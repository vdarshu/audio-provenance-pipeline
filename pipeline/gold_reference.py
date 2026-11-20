"""
Analytical modelling (Gold), part 1: load a reference batch into the
reference library tables (ref_track + ref_landmark) in Postgres.

The landmark table gets big (several million rows for FMA small), so it is
loaded with COPY and its index is rebuilt once at the end instead of being
maintained row by row.

    python -m pipeline.gold_reference --batch-id ref_fma_small
"""
import argparse

import pandas as pd

from pipeline import config, db
from pipeline.validate import quality_path


def _vector_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def load_reference(batch_id):
    quality = pd.read_parquet(quality_path(batch_id))
    feats = pd.read_parquet(config.SILVER_DIR / "features" / f"batch_id={batch_id}")
    feats = feats[feats["error"].isna()]
    tracks = feats.merge(quality, on="file_id", how="inner", suffixes=("", "_bronze"))

    with db.run_logged("gold_reference", batch_id) as run, db.cursor() as cur:
        # make the load idempotent: drop anything from a previous run of this batch
        cur.execute("""DELETE FROM ref_landmark WHERE track_id IN
                       (SELECT track_id FROM ref_track WHERE batch_id = %s)""", (batch_id,))
        cur.execute("DELETE FROM ref_track WHERE batch_id = %s", (batch_id,))

        db.copy_rows(
            cur, "ref_track",
            ["file_id", "batch_id", "rel_path", "sha256", "title", "artist",
             "licence", "duration_s", "n_hashes", "embedding"],
            (
                (r.file_id, batch_id, r.rel_path, r.sha256,
                 _clean(r.title), _clean(r.artist), _clean(r.licence),
                 round(float(r.duration_s), 3), int(r.n_hashes), _vector_literal(r.embedding))
                for r in tracks.itertuples()
            ),
        )
        cur.execute("SELECT file_id, track_id FROM ref_track WHERE batch_id = %s", (batch_id,))
        track_ids = dict(cur.fetchall())

        cur.execute("DROP INDEX IF EXISTS ix_ref_landmark_hash")
        n_landmarks = 0
        lm_dir = config.SILVER_DIR / "landmarks" / f"batch_id={batch_id}"
        for part in sorted(lm_dir.glob("*.parquet")):
            lm = pd.read_parquet(part)
            lm = lm[lm["file_id"].isin(track_ids)]
            lm["track_id"] = lm["file_id"].map(track_ids)
            db.copy_rows(cur, "ref_landmark", ["hash", "track_id", "t_offset"],
                         lm[["hash", "track_id", "t_offset"]].itertuples(index=False))
            n_landmarks += len(lm)

        cur.execute("CREATE INDEX ix_ref_landmark_hash ON ref_landmark (hash)")
        cur.execute("""CREATE INDEX IF NOT EXISTS ix_ref_track_embedding
                       ON ref_track USING hnsw (embedding vector_cosine_ops)""")
        cur.execute("ANALYZE ref_landmark")
        cur.execute("ANALYZE ref_track")
        cur.execute("UPDATE dim_batch SET status = 'reference_loaded' WHERE batch_id = %s",
                    (batch_id,))
        run["rows_in"], run["rows_out"] = len(feats), len(tracks)
        run["details"] = {"landmarks": n_landmarks}

    print(f"reference {batch_id}: {len(tracks)} tracks, {n_landmarks} landmarks loaded")


def _clean(value):
    """COPY text format can't take raw tabs/newlines, and pandas NaN must become NULL."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value).replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r", " ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    load_reference(ap.parse_args().batch_id)


if __name__ == "__main__":
    main()
