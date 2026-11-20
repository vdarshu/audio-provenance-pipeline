"""
Screens incoming vendor batches against the reference library.

Every 30 minutes the first task looks for a batch in the landing zone that has
not been screened yet. If there is none the run short-circuits and does
nothing; otherwise the batch goes through ingest -> validation -> fingerprints
-> screening, and a compliance report is written to lake/gold/reports/.

A batch can also be pushed through by hand:
    airflow dags trigger vendor_batch_screening --conf '{"batch_id": "vendor_synth_001"}'

SLA: a batch should be fully screened within 4 hours of landing
(docs/01_problem_framing_and_slas.md). The SLA is set on the screening task.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

PROJECT = "/opt/project"
CLI = f"cd {PROJECT} && python -m pipeline"
BATCH = "{{ ti.xcom_pull(task_ids='pick_batch', key='batch_id') }}"


def pick_batch(**context):
    """Return True (and push the batch id) if there is something to screen."""
    import sys
    sys.path.insert(0, PROJECT)
    from pipeline import db
    from pipeline.landing import get_landing

    requested = (context["dag_run"].conf or {}).get("batch_id")
    if requested:
        candidates = [requested]
    else:
        with db.cursor(commit=False) as cur:
            cur.execute("SELECT batch_id FROM dim_batch WHERE status IN "
                        "('screened', 'failed_quality')")
            done = {r[0] for r in cur.fetchall()}
        candidates = [b for b in get_landing().list_batches("vendor") if b not in done]

    if not candidates:
        print("no new vendor batches")
        return False
    context["ti"].xcom_push(key="batch_id", value=candidates[0])
    print(f"screening {candidates[0]} ({len(candidates) - 1} more waiting)")
    return True


default_args = {
    "owner": "data-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="vendor_batch_screening",
    description="Vendor audio batch -> Bronze -> QC -> Silver -> screening -> report",
    start_date=datetime(2026, 11, 1),
    schedule=timedelta(minutes=30),
    catchup=False,
    max_active_runs=1,       # one batch at a time keeps the hash lookups predictable
    default_args=default_args,
    tags=["audio", "compliance"],
) as dag:

    pick = ShortCircuitOperator(task_id="pick_batch", python_callable=pick_batch)

    ingest = BashOperator(
        task_id="ingest_to_bronze",
        bash_command=f"{CLI}.ingest --role vendor --batch-id {BATCH}",
    )
    validate = BashOperator(
        task_id="validate_bronze",
        bash_command=f"{CLI}.validate --batch-id {BATCH}",
    )
    silver = BashOperator(
        task_id="silver_fingerprints",
        bash_command=f"{CLI}.silver_features --batch-id {BATCH}",
        execution_timeout=timedelta(hours=2),
    )
    screen = BashOperator(
        task_id="screen_against_reference",
        bash_command=f"{CLI}.gold_screening --batch-id {BATCH}",
        sla=timedelta(hours=4),
    )
    # Only synthetic benchmark batches have an answer key; real deliveries skip this.
    evaluate = BashOperator(
        task_id="evaluate_if_ground_truth",
        bash_command=(
            f"if [ -f {PROJECT}/data/ground_truth/{BATCH}.csv ]; then "
            f"{CLI}.evaluate --batch-id {BATCH}; "
            "else echo 'no ground truth for this batch, skipping'; fi"
        ),
    )

    pick >> ingest >> validate >> silver >> screen >> evaluate
