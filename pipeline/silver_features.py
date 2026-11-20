"""
Transformation: Bronze -> Silver (PySpark).

Reads the validated Bronze files for one batch, decodes and fingerprints each
one in parallel across Spark executors, and writes two Parquet datasets:

    silver/features/batch_id=<id>/   one row per file: duration, 72-d embedding, timing
    silver/landmarks/batch_id=<id>/  one row per landmark hash: (file_id, hash, t_offset)

Decoding is the expensive part, so the work is distributed with mapPartitions
and each partition imports librosa once. Spark runs in local[*] mode by
default; pointing SPARK_MASTER at a cluster needs no code change as long as the
executors can see the Bronze path.

    python -m pipeline.silver_features --batch-id vendor_acme_2026_11_15
"""
import argparse
import os
import time

import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline import config, db
from pipeline.validate import quality_path

FEATURE_SCHEMA = T.StructType([
    T.StructField("file_id", T.StringType(), False),
    T.StructField("duration_s", T.DoubleType()),
    T.StructField("embedding", T.ArrayType(T.FloatType())),
    T.StructField("n_hashes", T.IntegerType()),
    T.StructField("extract_ms", T.DoubleType()),
    T.StructField("error", T.StringType()),
])

LANDMARK_SCHEMA = T.StructType([
    T.StructField("file_id", T.StringType(), False),
    T.StructField("hash", T.LongType(), False),
    T.StructField("t_offset", T.IntegerType(), False),
])


def spark_session(app_name="silver_features"):
    return (
        SparkSession.builder.master(os.getenv("SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _fingerprint_partition(rows):
    # imported here so the import happens on the executor, not the driver
    from pipeline.fingerprint import fingerprint_file

    for row in rows:
        start = time.perf_counter()
        try:
            fp = fingerprint_file(row.bronze_path)
            elapsed = (time.perf_counter() - start) * 1000
            yield ("feature", Row(file_id=row.file_id, duration_s=fp["duration_s"],
                                  embedding=fp["embedding"].tolist(),
                                  n_hashes=int(len(fp["hashes"])),
                                  extract_ms=elapsed, error=None))
            for h, t in zip(fp["hashes"].tolist(), fp["offsets"].tolist()):
                yield ("landmark", Row(file_id=row.file_id, hash=h, t_offset=t))
        except Exception as exc:
            # A file that passed QC but still fails here is recorded, not fatal.
            yield ("feature", Row(file_id=row.file_id, duration_s=None, embedding=None,
                                  n_hashes=0, extract_ms=None,
                                  error=f"{type(exc).__name__}: {exc}"[:300]))


def build_silver(batch_id):
    spark = spark_session()
    spark.sparkContext.setLogLevel("WARN")
    feat_out = config.SILVER_DIR / "features" / f"batch_id={batch_id}"
    lm_out = config.SILVER_DIR / "landmarks" / f"batch_id={batch_id}"

    with db.run_logged("silver_features", batch_id) as run:
        # Only two columns are needed here. Reading them with pandas avoids Spark 3.5
        # rejecting the nanosecond timestamps pandas writes elsewhere in the table.
        quality = pd.read_parquet(quality_path(batch_id),
                                  columns=["file_id", "bronze_path", "qc_status"])
        quality = quality[quality["qc_status"] == "pass"]
        valid = spark.createDataFrame(
            list(quality[["file_id", "bronze_path"]].itertuples(index=False, name=None)),
            "file_id string, bronze_path string",
        )
        n_valid = len(quality)

        # ~4 files per task keeps executors busy without huge per-task overhead
        n_parts = max(1, min(n_valid // 4, 400))
        tagged = valid.repartition(n_parts).rdd.mapPartitions(_fingerprint_partition)
        tagged.cache()

        features = spark.createDataFrame(
            tagged.filter(lambda kv: kv[0] == "feature").map(lambda kv: kv[1]), FEATURE_SCHEMA)
        landmarks = spark.createDataFrame(
            tagged.filter(lambda kv: kv[0] == "landmark").map(lambda kv: kv[1]), LANDMARK_SCHEMA)

        features.write.mode("overwrite").parquet(str(feat_out))
        landmarks.write.mode("overwrite").parquet(str(lm_out))

        stats = features.agg(
            F.count("*").alias("files"),
            F.sum(F.when(F.col("error").isNotNull(), 1).otherwise(0)).alias("errors"),
            F.sum("n_hashes").alias("hashes"),
            F.expr("percentile_approx(extract_ms, 0.95)").alias("p95_extract_ms"),
        ).first().asDict()
        tagged.unpersist()

        run["rows_in"], run["rows_out"] = n_valid, stats["files"] - stats["errors"]
        run["details"] = {k: (float(v) if v is not None else None) for k, v in stats.items()}

    print(f"silver for {batch_id}: {stats}")
    spark.stop()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    build_silver(ap.parse_args().batch_id)


if __name__ == "__main__":
    main()
