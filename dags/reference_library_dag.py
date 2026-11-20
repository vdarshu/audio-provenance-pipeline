"""
Builds (or rebuilds) the reference library from the public datasets.

Run manually after scripts/download_datasets.py has put the archives in data/raw.
Each source goes through the same Bronze -> quality -> Silver -> Gold steps
as a vendor delivery, so the reference data gets the same checks.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

PROJECT = "/opt/project"
CLI = f"cd {PROJECT} && python -m pipeline"

default_args = {
    "owner": "data-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="reference_library_build",
    description="Public datasets -> reference library (ref_track / ref_landmark)",
    start_date=datetime(2026, 11, 1),
    schedule=None,           # manual trigger only
    catchup=False,
    default_args=default_args,
    params={"fma_subset": "fma_small"},
    tags=["audio", "reference"],
) as dag:

    prepare = BashOperator(
        task_id="prepare_landing",
        bash_command=f"cd {PROJECT} && python scripts/prepare_landing.py "
                     "--fma-subset {{ params.fma_subset }}",
    )

    for batch in ["ref_{{ params.fma_subset }}", "ref_librispeech_dev_clean"]:
        # task ids can't contain template braces, so name groups by source family
        group_id = "fma" if "fma" in batch else "librispeech"
        with TaskGroup(group_id=group_id) as tg:
            ingest = BashOperator(
                task_id="ingest_to_bronze",
                bash_command=f"{CLI}.ingest --role reference --batch-id {batch}",
            )
            validate = BashOperator(
                task_id="validate_bronze",
                bash_command=f"{CLI}.validate --batch-id {batch}",
            )
            silver = BashOperator(
                task_id="silver_fingerprints",
                bash_command=f"{CLI}.silver_features --batch-id {batch}",
                execution_timeout=timedelta(hours=6),
            )
            gold = BashOperator(
                task_id="load_reference_library",
                bash_command=f"{CLI}.gold_reference --batch-id {batch}",
            )
            ingest >> validate >> silver >> gold
        prepare >> tg
