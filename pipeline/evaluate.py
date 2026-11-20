"""
Measure detection quality on a synthetic vendor batch.

scripts/make_vendor_batch.py writes a ground-truth CSV (kept outside the
landing zone, a real vendor would never send one). This joins it with
fact_screening and reports precision / recall / F1, recall per transformation,
whether the right reference track was identified, and latency percentiles.

A file counts as "predicted positive" if it was FLAGGED. REVIEW is reported
separately; treating REVIEW as positive gives the recall you'd get if every
grey-zone file went to a human.

    python -m pipeline.evaluate --batch-id vendor_synth_001
"""
import argparse
import json

import pandas as pd

from pipeline import config, db


def _prf(y_true, y_pred):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4)}


def evaluate(batch_id):
    truth = pd.read_csv(config.GROUND_TRUTH_DIR / f"{batch_id}.csv")
    with db.cursor(commit=False) as cur:
        cur.execute(
            """SELECT f.rel_path, f.decision, f.hash_votes, f.n_query_hashes, f.extract_ms, f.match_ms,
                      r.batch_id AS matched_batch, r.rel_path AS matched_rel_path
               FROM fact_screening f LEFT JOIN ref_track r ON r.track_id = f.best_track_id
               WHERE f.batch_id = %s""", (batch_id,))
        res = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])

    df = truth.merge(res, on="rel_path", how="left")
    # QC rejects (corrupt / silent / in-batch dupes) are excluded from detection metrics;
    # they are counted separately so nothing silently disappears.
    qc_rejects = df[df["decision"] == "REJECTED_QC"]
    # qc_defect rows are deliberately broken files; they test validation, not matching.
    scored = df[(df["decision"] != "REJECTED_QC") & (df["label"] != "qc_defect")].copy()

    y_true = scored["label"].eq("positive")
    flagged = scored["decision"].eq("FLAGGED")
    flagged_or_review = scored["decision"].isin(["FLAGGED", "REVIEW"])

    pos = scored[y_true]
    correct_source = (
        (pos["matched_batch"] == pos["source_batch"])
        & (pos["matched_rel_path"] == pos["source_rel_path"])
    )

    per_transform = (
        scored[y_true].assign(hit=flagged[y_true])
        .groupby("transform")["hit"].agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "recall"}).round(4)
    )
    # How F1 moves with the vote threshold - used to pick MATCH_MIN_VOTES.
    ratio = scored["hash_votes"] / scored["n_query_hashes"].where(scored["n_query_hashes"] > 0)
    sweep = []
    for votes in (5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100):
        pred = (scored["hash_votes"] >= votes) & (ratio >= config.MATCH_MIN_RATIO)
        sweep.append({"min_votes": votes, **_prf(y_true, pred.fillna(False))})
    best = max(sweep, key=lambda r: (r["f1"], r["precision"]))

    total_ms = scored["extract_ms"].fillna(0) + scored["match_ms"].fillna(0)

    report = {
        "batch_id": batch_id,
        "files_in_truth": len(truth),
        "scored": len(scored),
        "rejected_by_qc": len(qc_rejects),
        "qc_defects_caught": f"{int((qc_rejects['label'] == 'qc_defect').sum())}"
                             f"/{int((df['label'] == 'qc_defect').sum())}",
        "flagged_only": _prf(y_true, flagged),
        "flagged_or_review": _prf(y_true, flagged_or_review),
        "correct_source_identified": round(float(correct_source.mean()), 4) if len(pos) else None,
        "recall_by_transform": per_transform.to_dict(orient="index"),
        "threshold_sweep": sweep,
        "best_min_votes": best["min_votes"],
        "latency_ms": {
            "p50": round(float(total_ms.quantile(0.5)), 1),
            "p95": round(float(total_ms.quantile(0.95)), 1),
            "max": round(float(total_ms.max()), 1),
        },
    }
    out = config.GOLD_DIR / "reports" / f"{batch_id}_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    evaluate(ap.parse_args().batch_id)


if __name__ == "__main__":
    main()
