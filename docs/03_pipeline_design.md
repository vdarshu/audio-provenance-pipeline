# 3. Pipeline design

## Flow

```
 landing zone                 Bronze                   Silver                    Gold
 (local dir or MinIO)         lake/bronze/             lake/silver/              Postgres + lake/gold/
 ───────────────────          ──────────────           ──────────────            ─────────────────────────
 reference/<batch>/   ingest  audio/ (exact copy)  →   features/ (embedding,  →  ref_track, ref_landmark
 vendor/<batch>/     ──────►  manifest/ (sha256,       duration, timings)        fact_screening, dim_batch
                              lineage, licence)        landmarks/ (hash,         vw_batch_compliance
                              quality/ (QC result) ─►  t_offset per file)        reports/*.csv / *.json
                                        │                  (PySpark)             approved/  quarantine/
                                        └─ QC fail: kept in Bronze, excluded downstream
```

Two Airflow DAGs drive it:

| DAG | Trigger | Steps |
|---|---|---|
| `reference_library_build` | manual | prepare_landing → per source: ingest → validate → silver → load_reference_library |
| `vendor_batch_screening` | every 30 min (or manual with `batch_id`) | pick_batch → ingest → validate → silver → screen_against_reference → evaluate_if_ground_truth |

Every stage is also a plain CLI (`python -m pipeline.<stage> --batch-id ...`), so it can be run and debugged without Airflow. Airflow calls those same commands.

## Layers

**Bronze (`pipeline/ingest.py`, `pipeline/validate.py`)**
Byte-for-byte copy of the delivery plus a manifest: `file_id` (stable hash of batch + path), SHA-256, size, format, source, licence, ingest time. Bronze is never modified. The validation step adds a quality table with one row per file and a `qc_status` / `qc_reasons` pair, plus a batch quality report.

Quality rules: supported format, non-empty, decodable, duration 2-900 s, decoded length matches the header (catches truncated downloads), sample rate >= 8 kHz, at most 2 channels, not silent (RMS > -60 dBFS), not an in-batch duplicate, and licence present for reference data. Below 80% valid the whole batch stops.

**Silver (`pipeline/silver_features.py`, `pipeline/fingerprint.py`)**
PySpark job, one task per small group of files (`mapPartitions`). Each file is decoded to 11.025 kHz mono and produces:
- a 72-dimension embedding (MFCC, chroma and spectral-contrast mean/std, each block L2-normalised)
- landmark hashes: spectrogram peaks (at most 12 per second) paired with up to 5 later peaks within ~1.9 s; hash = (f1, f2, Δt) packed into 28 bits, stored with the anchor time

**Gold (`pipeline/gold_reference.py`, `pipeline/gold_screening.py`)**
Postgres tables (schema in `sql/01_lake_schema.sql`):

| Table / view | Grain | Purpose |
|---|---|---|
| `dim_batch` | batch | role, source, licence, landed/ingested/screened times, status |
| `ref_track` | reference file | metadata, licence, `embedding vector(72)` with HNSW cosine index |
| `ref_landmark` | hash occurrence | (hash, track_id, t_offset), B-tree index on hash |
| `fact_screening` | screened vendor file | QC result, best match, votes, offset, embedding similarity, decision, timings, run_id |
| `etl_run_log` | stage run | status, row counts, duration, error, details |
| `vw_batch_compliance` | batch | counts per decision, % clear, p95 latency |
| `vw_top_matched_works` | reference track | how often a protected work turns up in deliveries |

## How a match is decided

For an incoming clip, all of its hashes are looked up in `ref_landmark` and the hits are grouped by (reference track, reference time - clip time). A real copy lines up many hits at a single time difference; chance hits are scattered. The tallest group gives:

- `hash_votes`: the number of aligned hits
- `match_offset_s`: where in the reference track the clip starts, which is useful evidence for legal

FLAGGED needs votes >= `MATCH_MIN_VOTES` **and** votes / clip hashes >= `MATCH_MIN_RATIO`. REVIEW is the band below that. A byte-identical copy of an earlier file is always FLAGGED. The pgvector nearest neighbour is recorded alongside as a secondary "sounds similar" signal, but it does not decide, because two different songs in the same genre can have similar embeddings.

The thresholds in `pipeline/config.py` are starting values. `pipeline/evaluate.py` prints an F1 sweep over vote thresholds for each benchmark batch, and the thresholds are re-tuned from that.

## Known limitation

Landmark hashes rely on exact frequency positions, so **pitch-shifted** copies are not detected. Expect close to 0% recall for that alteration. It is kept in the benchmark on purpose and reported separately. Possible fixes for the final submission: pitch-invariant hashing (frequency ratios instead of absolute bins), or running the lookup on a few pitch-shifted versions of each incoming clip.

## Changes from the proposal

- **Vector store**: PostgreSQL + pgvector instead of "PostgreSQL / Milvus". One database holds metadata, hashes and vectors, which keeps the stack inside the 40-hour budget.
- **Primary metric**: "accuracy" replaced by F1 and precision, with recall reported per alteration type (see docs/01).
- **Ground truth**: synthetic vendor batches with an answer key, built from held-out data (see docs/02).
