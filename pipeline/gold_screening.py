"""
Analytical modelling (Gold), part 2: screen a vendor batch against the
reference library and decide what happens to every file.

For each file that made it to Silver:
  1. embedding nearest neighbour via pgvector (cosine) - cheap "sounds like" signal
  2. landmark hash lookup: count hash hits per (reference track, time-offset
     difference). Copies produce a tall spike at one offset; unrelated audio
     produces a few scattered hits. The spike height ("votes") drives the decision.

Decisions:
  FLAGGED      votes >= MATCH_MIN_VOTES and votes / clip hashes >= MATCH_MIN_RATIO,
               or byte-identical to an earlier file
  REVIEW       MATCH_REVIEW_VOTES <= votes < MATCH_MIN_VOTES  (human check)
  CLEAR        everything else
  REJECTED_QC  failed Bronze validation or could not be fingerprinted

CLEAR files are copied to lake/approved/, FLAGGED and REVIEW files to
lake/quarantine/. Results land in fact_screening plus a Parquet export and a
CSV/JSON compliance report for the legal team.

    python -m pipeline.gold_screening --batch-id vendor_acme_2026_11_15
"""
import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config, db
from pipeline.fingerprint import frames_to_seconds
from pipeline.gold_reference import _vector_literal
from pipeline.validate import quality_path

HASH_VOTE_SQL = """
    SELECT r.track_id, r.t_offset - q.t_offset AS delta, count(*) AS votes
    FROM unnest(%s::int[], %s::int[]) AS q(hash, t_offset)
    JOIN ref_landmark r ON r.hash = q.hash
    GROUP BY r.track_id, delta
    ORDER BY votes DESC
    LIMIT 1
"""

EMBED_SQL = """
    SELECT track_id, 1 - (embedding <=> %s::vector) AS similarity
    FROM ref_track
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""


def match_one(cur, embedding, hashes, offsets):
    start = time.perf_counter()
    out = {"embed_track_id": None, "embed_similarity": None, "best_track_id": None,
           "hash_votes": 0, "match_offset_s": None, "n_query_hashes": len(hashes)}

    vec = _vector_literal(embedding)
    cur.execute(EMBED_SQL, (vec, vec, config.EMBED_TOP_K))
    top = cur.fetchone()
    if top:
        out["embed_track_id"], out["embed_similarity"] = top[0], float(top[1])

    if len(hashes):
        cur.execute(HASH_VOTE_SQL, (list(map(int, hashes)), list(map(int, offsets))))
        best = cur.fetchone()
        if best:
            out["best_track_id"], delta, out["hash_votes"] = best[0], best[1], int(best[2])
            out["match_offset_s"] = round(float(frames_to_seconds(delta)), 2)

    out["match_ms"] = (time.perf_counter() - start) * 1000
    return out


def decide(row):
    if row["qc_status"] != "pass" or pd.notna(row.get("silver_error")):
        return "REJECTED_QC"
    if row.get("exact_copy_of_earlier"):
        return "FLAGGED"
    # The ratio guard stops long clips from being flagged on a pile of chance hits:
    # a real copy lines up a meaningful share of its hashes, not just a raw count.
    ratio = row["hash_votes"] / row["n_query_hashes"] if row.get("n_query_hashes") else 0
    if row["hash_votes"] >= config.MATCH_MIN_VOTES and ratio >= config.MATCH_MIN_RATIO:
        return "FLAGGED"
    if row["hash_votes"] >= config.MATCH_REVIEW_VOTES:
        return "REVIEW"
    return "CLEAR"


def route_files(df, batch_id):
    """Physically separate approved audio from quarantined audio."""
    for row in df.itertuples():
        if row.decision == "CLEAR":
            dest = config.APPROVED_DIR / batch_id / row.rel_path
        elif row.decision in ("FLAGGED", "REVIEW"):
            dest = config.QUARANTINE_DIR / batch_id / row.decision.lower() / row.rel_path
        else:
            continue   # QC rejects stay only in Bronze
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.bronze_path, dest)


def screen_batch(batch_id, route=True):
    quality = pd.read_parquet(quality_path(batch_id))
    feats = pd.read_parquet(config.SILVER_DIR / "features" / f"batch_id={batch_id}")
    feats = feats.rename(columns={"error": "silver_error", "duration_s": "fp_duration_s"})
    landmarks = pd.read_parquet(config.SILVER_DIR / "landmarks" / f"batch_id={batch_id}")
    lm_by_file = {fid: g for fid, g in landmarks.groupby("file_id")}

    df = quality.merge(feats, on="file_id", how="left")

    with db.run_logged("gold_screening", batch_id) as run:
        matches = []
        with db.cursor(commit=False) as cur:
            for row in df.itertuples():
                if row.qc_status != "pass" or not isinstance(row.embedding, np.ndarray):
                    matches.append({})
                    continue
                lm = lm_by_file.get(row.file_id)
                hashes = lm["hash"].to_numpy() if lm is not None else np.array([])
                offsets = lm["t_offset"].to_numpy() if lm is not None else np.array([])
                matches.append(match_one(cur, row.embedding, hashes, offsets))

        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(matches)], axis=1)
        df["hash_votes"] = df["hash_votes"].fillna(0).astype(int)
        df["decision"] = df.apply(decide, axis=1)
        df["run_id"] = run["run_id"]
        df["screened_at"] = datetime.now(timezone.utc)

        _write_fact(df, batch_id, run["run_id"])
        _export(df, batch_id)
        if route:
            route_files(df, batch_id)

        counts = df["decision"].value_counts().to_dict()
        run["rows_in"], run["rows_out"] = len(df), int(counts.get("CLEAR", 0))
        run["details"] = counts

    print(f"screened {batch_id}: {counts}")
    return df


def _int_or_none(v):
    return None if pd.isna(v) else int(v)


def _float_or_none(v, nd=3):
    return None if pd.isna(v) else round(float(v), nd)


def _write_fact(df, batch_id, run_id):
    with db.cursor() as cur:
        cur.execute("DELETE FROM fact_screening WHERE batch_id = %s", (batch_id,))
        db.copy_rows(
            cur, "fact_screening",
            ["batch_id", "file_id", "rel_path", "sha256", "qc_status", "qc_reasons",
             "best_track_id", "hash_votes", "n_query_hashes", "match_offset_s",
             "embed_track_id", "embed_similarity", "decision", "extract_ms", "match_ms",
             "run_id"],
            (
                (batch_id, r.file_id, r.rel_path, r.sha256, r.qc_status, r.qc_reasons or None,
                 _int_or_none(r.best_track_id), int(r.hash_votes),
                 _int_or_none(r.n_query_hashes), _float_or_none(r.match_offset_s, 2),
                 _int_or_none(r.embed_track_id), _float_or_none(r.embed_similarity, 4),
                 r.decision, _float_or_none(r.extract_ms, 1), _float_or_none(r.match_ms, 1),
                 run_id)
                for r in df.itertuples()
            ),
        )
        cur.execute("UPDATE dim_batch SET screened_at = now(), status = 'screened' "
                    "WHERE batch_id = %s", (batch_id,))


def _export(df, batch_id):
    """Gold Parquet for analysts + a flat report the legal team can open in Excel."""
    out_dir = config.GOLD_DIR / "screening" / f"batch_id={batch_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["batch_id", "file_id", "rel_path", "sha256", "qc_status", "qc_reasons",
            "decision", "best_track_id", "hash_votes", "n_query_hashes", "match_offset_s",
            "embed_track_id", "embed_similarity", "extract_ms", "match_ms", "screened_at"]
    df[keep].to_parquet(out_dir / "screening.parquet", index=False)

    with db.cursor(commit=False) as cur:
        cur.execute("SELECT * FROM vw_batch_compliance WHERE batch_id = %s", (batch_id,))
        cols = [c.name for c in cur.description]
        summary = dict(zip(cols, cur.fetchone()))
        cur.execute(
            """SELECT f.rel_path, f.decision, f.hash_votes, f.match_offset_s,
                      r.title, r.artist, r.licence, r.batch_id AS reference_source
               FROM fact_screening f LEFT JOIN ref_track r ON r.track_id = f.best_track_id
               WHERE f.batch_id = %s AND f.decision IN ('FLAGGED', 'REVIEW')
               ORDER BY f.hash_votes DESC""", (batch_id,))
        flagged = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])

    report_dir = config.GOLD_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(report_dir / f"{batch_id}_flagged_files.csv", index=False)
    (report_dir / f"{batch_id}_compliance_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--no-route", action="store_true", help="don't copy files to approved/quarantine")
    args = ap.parse_args()
    screen_batch(args.batch_id, route=not args.no_route)


if __name__ == "__main__":
    main()
