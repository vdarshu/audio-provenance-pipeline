"""
Quality validation on Bronze.

Each file gets a pass/fail plus a list of reasons. Files that fail are not
deleted - they stay in Bronze for the audit trail and are simply excluded from
Silver. The batch as a whole is stopped if too many files fail, because that
usually means the vendor delivery is broken rather than a few bad files.

Checks (per file):
  - extension is an audio format we accept
  - file is not empty and can be decoded
  - duration within [QC_MIN_DURATION_S, QC_MAX_DURATION_S]
  - decoded length matches the header (catches truncated downloads, whose
    header still claims the full duration)
  - native sample rate >= QC_MIN_SAMPLE_RATE, 1-2 channels
  - not silent (RMS above QC_SILENCE_DBFS)
  - not an exact duplicate (sha256) of another file in the same batch
  - reference batches only: licence is known

An exact byte copy of a file from an earlier batch is NOT a quality failure -
it is recorded in `exact_copy_of_earlier` and the gold layer treats it as the
strongest possible match evidence.

    python -m pipeline.validate --batch-id vendor_acme_2026_11_15
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from pipeline import config, db
from pipeline.ingest import manifest_path


def quality_path(batch_id):
    return config.BRONZE_DIR / "quality" / f"batch_id={batch_id}" / "quality.parquet"


def probe(path):
    """Decode-level checks for one file. Runs in a worker process."""
    result = {"bronze_path": path, "duration_s": None, "decoded_s": None, "sample_rate": None,
              "channels": None, "rms_dbfs": None, "decode_error": None}
    try:
        try:
            info = sf.info(path)
            result.update(sample_rate=info.samplerate, channels=info.channels,
                          duration_s=info.duration)
        except Exception:
            # some formats (m4a) aren't handled by libsndfile; librosa falls back to audioread
            result["duration_s"] = librosa.get_duration(path=path)

        # 30 s is enough to judge loudness without decoding hour-long files
        y, _ = librosa.load(path, sr=config.FP_SAMPLE_RATE, mono=True, duration=30)
        result["decoded_s"] = len(y) / config.FP_SAMPLE_RATE
        if y.size == 0:
            result["decode_error"] = "no samples decoded"
        else:
            rms = float(np.sqrt(np.mean(y ** 2)))
            result["rms_dbfs"] = 20 * np.log10(rms) if rms > 0 else -120.0
    except Exception as exc:
        result["decode_error"] = type(exc).__name__ + ": " + str(exc)[:200]
    return result


def previously_seen_hashes(current_batch):
    """sha256 values already present in earlier Bronze batches."""
    seen = set()
    for p in (config.BRONZE_DIR / "manifest").glob("batch_id=*/manifest.parquet"):
        if p.parent.name != f"batch_id={current_batch}":
            seen.update(pd.read_parquet(p, columns=["sha256"])["sha256"])
    return seen


def rule_reasons(row, role):
    reasons = []
    if row["extension"] not in config.AUDIO_EXTENSIONS:
        reasons.append("unsupported_format")
    if row["size_bytes"] == 0:
        reasons.append("empty_file")
    if pd.notna(row["decode_error"]):
        reasons.append("decode_error")
    else:
        dur = row["duration_s"] if pd.notna(row["duration_s"]) else 0
        if dur < config.QC_MIN_DURATION_S:
            reasons.append("too_short")
        if dur > config.QC_MAX_DURATION_S:
            reasons.append("too_long")
        # only the first 30 s are decoded, so compare against min(header, 30)
        expected = min(dur, 30.0)
        if pd.notna(row["decoded_s"]) and row["decoded_s"] < 0.9 * expected:
            reasons.append("truncated")
        if pd.notna(row["sample_rate"]) and row["sample_rate"] < config.QC_MIN_SAMPLE_RATE:
            reasons.append("low_sample_rate")
        if pd.notna(row["channels"]) and row["channels"] > 2:
            reasons.append("too_many_channels")
        if pd.notna(row["rms_dbfs"]) and row["rms_dbfs"] < config.QC_SILENCE_DBFS:
            reasons.append("silent")
    if row["dup_in_batch"]:
        reasons.append("duplicate_in_batch")
    licence = row.get("licence")
    if role == "reference" and (pd.isna(licence) or licence in ("", "unknown")):
        reasons.append("missing_licence")
    return reasons


def validate_batch(batch_id):
    manifest = pd.read_parquet(manifest_path(batch_id))
    role = manifest["role"].iloc[0]

    with db.run_logged("validate", batch_id) as run:
        with ProcessPoolExecutor(max_workers=config.WORKERS) as pool:
            probes = pd.DataFrame(list(pool.map(probe, manifest["bronze_path"], chunksize=8)))
        df = manifest.merge(probes, on="bronze_path", how="left")

        # keep the first copy of an in-batch duplicate, flag the rest
        df["dup_in_batch"] = df.duplicated("sha256", keep="first")
        df["exact_copy_of_earlier"] = df["sha256"].isin(previously_seen_hashes(batch_id))

        reasons = df.apply(lambda r: rule_reasons(r, role), axis=1)
        df["qc_reasons"] = reasons.map(lambda r: ";".join(r))
        df["qc_status"] = np.where(reasons.map(len) == 0, "pass", "fail")

        out = quality_path(batch_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

        valid = int((df["qc_status"] == "pass").sum())
        ratio = valid / len(df)
        reason_counts = (df["qc_reasons"].str.split(";").explode()
                         .replace("", np.nan).dropna().value_counts().to_dict())
        report = {
            "batch_id": batch_id,
            "files": len(df),
            "valid": valid,
            "valid_ratio": round(ratio, 4),
            "failure_reasons": reason_counts,
            "total_duration_h": round(float(df["duration_s"].fillna(0).sum()) / 3600, 2),
            "sample_rates": df["sample_rate"].value_counts().to_dict(),
        }
        (out.parent / "quality_report.json").write_text(json.dumps(report, indent=2, default=str))

        with db.cursor() as cur:
            cur.execute("UPDATE dim_batch SET valid_count = %s, status = %s WHERE batch_id = %s",
                        (valid, "validated" if ratio >= config.QC_MIN_VALID_RATIO
                         else "failed_quality", batch_id))
        run["rows_in"], run["rows_out"], run["details"] = len(df), valid, report

    print(json.dumps(report, indent=2, default=str))
    if ratio < config.QC_MIN_VALID_RATIO:
        raise SystemExit(f"batch {batch_id} failed quality gate: "
                         f"{ratio:.1%} valid < {config.QC_MIN_VALID_RATIO:.0%}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    validate_batch(ap.parse_args().batch_id)


if __name__ == "__main__":
    main()
