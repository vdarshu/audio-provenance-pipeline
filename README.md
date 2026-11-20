# Audio fingerprinting and plagiarism screening pipeline

Capstone project, Executive PG Program in Data Science (Data Engineering track).

Vendor audio batches are screened before they enter an AI training data lake. Each file is fingerprinted from its spectrogram and matched against a reference library of known, licensed audio. Copies are flagged even when they have been re-encoded, resampled, trimmed or had noise added, which file hashes and metadata checks cannot do.

Stack: Python, PySpark, librosa, PostgreSQL 16 + pgvector, Apache Airflow 2.10, MinIO, Docker Compose.

## Documentation

| Doc | Contents |
|---|---|
| [docs/01_problem_framing_and_slas.md](docs/01_problem_framing_and_slas.md) | Problem, stakeholders, success metrics, data SLAs |
| [docs/02_data_sourcing_and_provenance.md](docs/02_data_sourcing_and_provenance.md) | Datasets, licences, download and checksum record, PII check, how ground truth is built |
| [docs/03_pipeline_design.md](docs/03_pipeline_design.md) | Layers, quality rules, gold data model, match logic, known limitations |

## Repository layout

```
dags/
  reference_library_dag.py     build the reference library from public datasets
  vendor_screening_dag.py      screen new vendor batches every 30 min
pipeline/
  config.py                    paths, thresholds, connection settings (env-overridable)
  landing.py                   landing zone access, local folder or MinIO/S3
  ingest.py                    landing -> Bronze (copy, sha256, manifest, lineage)
  source_metadata.py           FMA / LibriSpeech metadata and licences
  validate.py                  Bronze quality checks and batch quality gate
  fingerprint.py               embeddings + landmark hashes
  silver_features.py           PySpark job: Bronze -> Silver Parquet
  gold_reference.py            load reference library into Postgres
  gold_screening.py            match vendor files, decide, route, report
  evaluate.py                  precision / recall / F1 against ground truth
  db.py                        Postgres helpers and run log
scripts/
  download_datasets.py         download FMA + LibriSpeech, record checksums
  prepare_landing.py           split into reference batches + holdout
  make_vendor_batch.py         build a synthetic vendor batch with an answer key
sql/                           schema (runs on first Postgres start)
docker/                        Airflow image (adds Java, ffmpeg, libsndfile)
tests/                         fingerprint sanity tests
data/  lake/                   created at run time (not in git)
```

## Running it

### 1. Start the stack (Windows: Docker Desktop with the WSL 2 backend, give it at least 8 GB RAM)

```powershell
git clone <repo-url>
cd audio-provenance-pipeline
docker compose up -d --build
```

Airflow: http://localhost:8080 (admin / admin). MinIO console: http://localhost:9001 (minioadmin / minioadmin).

### 2. Get the data

```powershell
docker compose exec airflow-scheduler python /opt/project/scripts/download_datasets.py --sample 750
```

This fetches about 1 GB into `data/raw/`: 750 genre-balanced FMA tracks plus `tracks.csv`, read straight out of the remote zips, and LibriSpeech `dev-clean`. It writes `data/raw/download_manifest.json` and `data/raw/fma_small_sample_files.csv`.

Drop `--sample` to download the full `fma_small` (7.6 GB in total). If Python's zip extraction fails on the full FMA archive, extract it with 7-Zip into `data/raw/`.

### 3. Build the reference library

In the Airflow UI, unpause and trigger **reference_library_build**. Or run the stages by hand:

```bash
python scripts/prepare_landing.py
for b in ref_fma_small ref_librispeech_dev_clean; do
  python -m pipeline.ingest --role reference --batch-id $b
  python -m pipeline.validate --batch-id $b
  python -m pipeline.silver_features --batch-id $b
  python -m pipeline.gold_reference --batch-id $b
done
```

### 4. Screen a vendor batch

```bash
python scripts/make_vendor_batch.py --batch-id vendor_synth_001 --positives 300 --negatives 300
```

Then unpause **vendor_batch_screening**. It picks up the new batch on its next run, or trigger it with `{"batch_id": "vendor_synth_001"}`. Results:

- `lake/gold/reports/vendor_synth_001_compliance_summary.json`: counts per decision, latency
- `lake/gold/reports/vendor_synth_001_flagged_files.csv`: flagged files with the matched work and its licence
- `lake/gold/reports/vendor_synth_001_evaluation.json`: precision / recall / F1, recall per alteration, threshold sweep
- `lake/approved/` and `lake/quarantine/`: routed audio
- Postgres `audio_lake`: `fact_screening`, `vw_batch_compliance`, `etl_run_log`

### Using MinIO as the landing zone

Upload batches to the `landing` bucket using the same layout (`vendor/<batch_id>/...` with a `_batch.json`) and set `LANDING_BACKEND=minio` before `docker compose up`.

### Running without Docker

Needs Python 3.11, Java 17 (for Spark), ffmpeg and a Postgres 16 with pgvector. Then run `pip install -r requirements.txt`, run `sql/*.sql` against an `audio_lake` database, and set `PG_DSN` if it isn't on localhost. Spark on native Windows also needs `winutils.exe`, so Docker or WSL is the easier route.

## Mid-submission checklist

| Task | Where |
|---|---|
| Problem framing and data SLAs | `docs/01_problem_framing_and_slas.md` |
| Data sourcing and provenance | `docs/02_...`, `scripts/download_datasets.py`, `data/raw/download_manifest.json` |
| Extraction / ingestion | `pipeline/landing.py`, `pipeline/ingest.py`, `scripts/prepare_landing.py` |
| Raw (Bronze) storage | `lake/bronze/audio`, `lake/bronze/manifest` |
| Quality validation | `pipeline/validate.py`, `lake/bronze/quality/*/quality_report.json` |
| Transformation (Silver) | `pipeline/fingerprint.py`, `pipeline/silver_features.py` |
| Analytical modelling (Gold) | `sql/01_lake_schema.sql`, `pipeline/gold_reference.py`, `pipeline/gold_screening.py` |
| Orchestration | `dags/`, `docker-compose.yml` |

## Still to do for the final submission

- Tune thresholds on the full benchmark and report final metrics
- Pitch-invariant matching (the main gap; see docs/03)
- Scale test on the full `fma_small` / `fma_medium` (the mid-submission runs on a 750-track sample): Silver throughput and hash-lookup latency
- Technical report and executive presentation

## Mid-submission results (750-track FMA sample + LibriSpeech dev-clean)

Reference library: 3,217 tracks (700 FMA, 2,517 LibriSpeech) and 2.27 M landmark hashes.
Vendor benchmark `vendor_synth_001`: 356 files (200 altered copies, 150 unseen clips, 6 broken files).

| Metric | Result | Target |
|---|---|---|
| F1 (FLAGGED) | 0.936 | >= 0.92 |
| Precision | 1.00 (0 false flags) | >= 0.95 |
| Recall | 0.88 | per transform |
| QC defects caught | 6 / 6 | all |
| Screening latency p50 / p95 | 0.40 s / 1.85 s | p95 < 2 s |
| Batch landed to screened | about 4 min | < 4 h |

Recall is 100% for exact copies, MP3 re-encodes, 8 kHz resampling, 10 s excerpts, gain changes and noise at 20 and 10 dB SNR. All 24 misses are +1 semitone pitch shifts (0% recall), the known limitation of absolute-frequency landmark hashes (docs/03). F1 stays the same for vote thresholds from 8 to 50, so the result doesn't depend on fine threshold tuning.

Evidence is in `results/`: Bronze quality reports, Gold reports and table exports, provenance files and Airflow screenshots.
