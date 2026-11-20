"""
Extraction / ingestion: landing zone -> Bronze.

Bronze is an exact, immutable copy of what was delivered plus a manifest with
lineage columns. Nothing is decoded or cleaned here; if a vendor sent a broken
file, Bronze keeps the broken file so we can prove what we received.

    python -m pipeline.ingest --role vendor --batch-id vendor_acme_2026_11_15
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import config, db, source_metadata
from pipeline.landing import get_landing


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_id_for(batch_id, rel_path):
    # deterministic, so re-running ingest gives the same ids downstream
    return hashlib.sha1(f"{batch_id}/{rel_path}".encode()).hexdigest()[:16]


def manifest_path(batch_id):
    return config.BRONZE_DIR / "manifest" / f"batch_id={batch_id}" / "manifest.parquet"


def ingest_batch(role, batch_id, force=False):
    out_manifest = manifest_path(batch_id)
    if out_manifest.exists() and not force:
        print(f"{batch_id} already in Bronze, skipping (use --force to redo)")
        return pd.read_parquet(out_manifest)

    landing = get_landing()
    meta = landing.batch_meta(role, batch_id)
    source = meta.get("source", batch_id)
    audio_root = config.BRONZE_DIR / "audio" / f"batch_id={batch_id}"
    ingested_at = datetime.now(timezone.utc)

    with db.run_logged("ingest", batch_id) as run:
        rows = []
        for rel_path in landing.list_files(role, batch_id):
            dest = audio_root / rel_path
            landing.fetch(role, batch_id, rel_path, dest)
            rows.append({
                "file_id": file_id_for(batch_id, rel_path),
                "batch_id": batch_id,
                "role": role,
                "source": source,
                "rel_path": rel_path,
                "bronze_path": str(dest),
                "extension": Path(rel_path).suffix.lower(),
                "size_bytes": dest.stat().st_size,
                "sha256": sha256_of(dest),
                "ingested_at": ingested_at,
            })
        if not rows:
            raise RuntimeError(f"batch {batch_id} is empty in the landing zone")

        df = pd.DataFrame(rows)
        df = source_metadata.enrich(df, source, default_licence=meta.get("licence"))
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_manifest, index=False, coerce_timestamps="us",
                      allow_truncated_timestamps=True)

        with db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dim_batch (batch_id, role, source, licence, landed_at,
                                       ingested_at, file_count, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'ingested')
                ON CONFLICT (batch_id) DO UPDATE
                   SET ingested_at = EXCLUDED.ingested_at,
                       file_count  = EXCLUDED.file_count,
                       status      = 'ingested'
                """,
                (batch_id, role, source, meta.get("licence"),
                 meta.get("landed_at", ingested_at.isoformat()), ingested_at, len(df)),
            )
        run["rows_in"] = run["rows_out"] = len(df)
        run["details"] = {"source": source, "bytes": int(df["size_bytes"].sum())}

    print(f"ingested {len(df)} files for {batch_id} "
          f"({df['size_bytes'].sum() / 1e6:.1f} MB)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=["reference", "vendor"], required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ingest_batch(args.role, args.batch_id, args.force)


if __name__ == "__main__":
    main()
