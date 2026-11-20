"""
Central settings for the pipeline.

Everything can be overridden with environment variables so the same code runs
on a laptop, inside the Airflow container, or in CI without edits.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"                 # original downloaded archives (kept untouched)
LANDING_DIR = DATA_DIR / "landing"         # where vendors / sources drop batches
HOLDOUT_DIR = DATA_DIR / "holdout"         # originals kept OUT of the reference library
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

LAKE_DIR = Path(os.getenv("LAKE_DIR", PROJECT_ROOT / "lake"))
BRONZE_DIR = LAKE_DIR / "bronze"
SILVER_DIR = LAKE_DIR / "silver"
GOLD_DIR = LAKE_DIR / "gold"
QUARANTINE_DIR = LAKE_DIR / "quarantine"
APPROVED_DIR = LAKE_DIR / "approved"

# Landing zone backend: "local" reads data/landing, "minio" reads an S3 bucket.
LANDING_BACKEND = os.getenv("LANDING_BACKEND", "local")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_LANDING_BUCKET = os.getenv("MINIO_LANDING_BUCKET", "landing")

PG_DSN = os.getenv(
    "PG_DSN", "host=localhost port=5432 dbname=audio_lake user=pipeline password=pipeline"
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Fingerprint parameters. 11 kHz mono is plenty for landmark hashing and keeps
# the STFT cheap; most of the identifying energy in music/speech sits below 5 kHz.
FP_SAMPLE_RATE = 11025
FP_N_FFT = 2048
FP_HOP = 512                 # ~46 ms per frame at 11025 Hz
FP_PEAK_NEIGHBOURHOOD = (15, 15)   # (freq bins, frames) for local-max search
FP_PEAKS_PER_SECOND = 12     # cap on peak density so long files don't explode the index
FP_FAN_OUT = 5               # how many targets each anchor peak is paired with
FP_MAX_DT = 40               # max frame gap between anchor and target (~1.9 s)
FP_MAX_SECONDS = 120         # only fingerprint the first N seconds of very long files

# Match decision thresholds. Starting values; re-tune with the threshold sweep
# printed by pipeline.evaluate once the real reference library is loaded.
MATCH_MIN_VOTES = int(os.getenv("MATCH_MIN_VOTES", 20))        # aligned hash hits to flag
MATCH_MIN_RATIO = float(os.getenv("MATCH_MIN_RATIO", 0.02))    # ...as a share of the clip's hashes
MATCH_REVIEW_VOTES = int(os.getenv("MATCH_REVIEW_VOTES", 10))  # grey zone -> manual review
EMBED_TOP_K = 5

# Data quality rules for Bronze validation.
QC_MIN_DURATION_S = 2.0
QC_MAX_DURATION_S = 900.0
QC_MIN_SAMPLE_RATE = 8000
QC_SILENCE_DBFS = -60.0
QC_MIN_VALID_RATIO = float(os.getenv("QC_MIN_VALID_RATIO", 0.80))  # hard stop below this

WORKERS = int(os.getenv("PIPELINE_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
