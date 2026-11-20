"""Small Postgres helpers shared by the gold layer and the run log."""
import contextlib
import io
import json
import time
import uuid

import psycopg2
import psycopg2.extras

from pipeline import config


def connect():
    return psycopg2.connect(config.PG_DSN)


@contextlib.contextmanager
def cursor(commit=True):
    conn = connect()
    try:
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    finally:
        conn.close()


def copy_rows(cur, table, columns, rows):
    """Bulk load with COPY - much faster than executemany for millions of hash rows."""
    buf = io.StringIO()
    for row in rows:
        buf.write("\t".join("\\N" if v is None else str(v) for v in row))
        buf.write("\n")
    buf.seek(0)
    cur.copy_expert(
        f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT text)", buf
    )


@contextlib.contextmanager
def run_logged(stage, batch_id):
    """
    Record every stage execution in etl_run_log so the audit trail shows what ran,
    when, on which batch and with what row counts. Usage:

        with run_logged("ingest", batch_id) as run:
            ...
            run["rows_in"] = 10
    """
    run = {"run_id": str(uuid.uuid4()), "rows_in": None, "rows_out": None, "details": {}}
    started = time.time()
    status, error = "success", None
    try:
        yield run
    except Exception as exc:  # log and re-raise so Airflow still sees the failure
        status, error = "failed", repr(exc)[:2000]
        raise
    finally:
        try:
            with cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO etl_run_log
                        (run_id, stage, batch_id, status, rows_in, rows_out,
                         started_at, finished_at, error, details)
                    VALUES (%s, %s, %s, %s, %s, %s, to_timestamp(%s), now(), %s, %s)
                    """,
                    (run["run_id"], stage, batch_id, status, run["rows_in"],
                     run["rows_out"], started, error, json.dumps(run["details"])),
                )
        except psycopg2.Error as db_err:
            # Logging must never hide the real result of the stage.
            print(f"[warn] could not write run log: {db_err}")
