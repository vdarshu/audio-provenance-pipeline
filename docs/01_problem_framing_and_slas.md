# 1. Problem framing and data SLAs

## The problem

Teams building audio AI models buy training data in bulk from outside vendors and pull it from open repositories. Some of what arrives is not what the vendor says it is:

- the same recording delivered twice, or already in the lake from an earlier batch
- copyrighted recordings passed off as licensed, usually altered slightly: re-encoded at a lower bitrate, resampled, cut to a shorter clip, volume-changed or with noise added

Training a commercial model on that data is a legal risk (copyright claims, forced model retraining) and a waste of storage and compute. Today this is checked by hand on samples, or with file hashes and metadata. Neither works:

| Current check | Why it fails |
|---|---|
| Manual spot checks | A batch is thousands of files; only a small sample is ever listened to |
| MD5 / SHA-256 of the file | Any re-encode, resample or trim gives a completely different hash |
| Title / artist metadata | Vendors strip or rewrite tags; metadata proves nothing |

## What the pipeline does

Every incoming batch is screened automatically against a **reference library** of audio whose owner and licence are known. The match is done on the sound itself (spectrogram landmark fingerprints), not on bytes or tags. Each file ends up with a decision, and the batch gets a compliance report:

| Decision | Meaning | Where the file goes |
|---|---|---|
| CLEAR | no meaningful match to the reference library | `lake/approved/` |
| REVIEW | weak match, needs a human to listen | `lake/quarantine/<batch>/review/` |
| FLAGGED | strong match or byte-identical copy | `lake/quarantine/<batch>/flagged/` |
| REJECTED_QC | broken, silent, truncated or duplicated file | stays in Bronze only |

## Stakeholders and the questions they need answered

| Stakeholder | Question | Answered by |
|---|---|---|
| Data sourcing / procurement | Which vendor batches are clean enough to accept or pay for? | `vw_batch_compliance`, batch compliance JSON |
| Legal and compliance | Which files match which protected works, and can we prove it? | `*_flagged_files.csv`, `fact_screening` + `ref_track` (match offset, votes, licence) |
| AI research lead | Which files are safe to train on? | `lake/approved/` |
| Data engineering | Did every stage run, on what, and how long did it take? | `etl_run_log`, Airflow |

## Success metrics

The proposal stated "over 92% detection accuracy". With far more clean files than copies, plain accuracy can look good while missing most copies, so the primary metric is now **F1 on the FLAGGED decision**, measured on synthetic vendor batches where the answer is known (see `docs/02` for how they are built).

| Metric | Target | Measured by |
|---|---|---|
| F1 for copied / altered audio (primary) | >= 0.92 | `pipeline/evaluate.py` |
| Precision (share of FLAGGED files that really are copies) | >= 0.95 | `pipeline/evaluate.py` |
| Recall per alteration type | reported for each type, not hidden in an average | `recall_by_transform` in the evaluation report |
| Correct source track identified for a flagged copy | >= 0.90 | `correct_source_identified` |
| Per-clip screening latency (fingerprint + lookup), p95 | < 2 s | `extract_ms + match_ms` in `fact_screening` |
| Batches processed by the DAG without manual steps | 100% | Airflow run history |

## Data SLAs

These apply to each vendor batch from the moment it lands.

| Dimension | SLA | How it is enforced / checked |
|---|---|---|
| Freshness | Batch fully screened within **4 hours** of landing | Airflow SLA on `screen_against_reference`; `dim_batch.landed_at` vs `screened_at` |
| Completeness | Every landed file has exactly one row in `fact_screening` | Row counts: manifest = quality table = `fact_screening`; logged in `etl_run_log` |
| Validity | >= 95% of files pass QC is expected; below **80%** the batch is stopped as `failed_quality` and returned to the vendor | `pipeline/validate.py` quality gate |
| Uniqueness | No exact duplicate (same SHA-256) reaches `approved/` | in-batch duplicates rejected at QC; copies of earlier files FLAGGED |
| Accuracy | F1 >= 0.92 on the benchmark batch; re-measured whenever thresholds or fingerprint settings change | `pipeline/evaluate.py` |
| Latency | p95 per-clip screening < 2 s | `vw_batch_compliance.p95_latency_ms` |
| Lineage / audit | Every decision traceable to batch, file SHA-256, reference track and pipeline run | `fact_screening.run_id`, `etl_run_log`, Bronze kept immutable |
| Retention | Bronze copies of every delivery are kept unchanged, including rejected files | ingest never modifies or deletes Bronze |

When an SLA is missed, Airflow marks the task (or SLA miss) and the batch stays out of `approved/` until it is rerun. A batch is never partly approved because a stage failed halfway.
