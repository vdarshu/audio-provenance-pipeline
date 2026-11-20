"""
Turn the extracted public datasets into reference batches in the landing zone.

A small, deterministic slice of each dataset (5% by default) is held back in
data/holdout/ and never enters the reference library. Those files are the
"genuinely new" audio used as negatives when building synthetic vendor batches -
without them we could only measure recall, not false positives.

Files are hard-linked where possible so 8 GB of audio isn't duplicated on disk.

    python scripts/prepare_landing.py
    python scripts/prepare_landing.py --fma-subset fma_medium --holdout 0.05
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config  # noqa: E402


def is_holdout(rel_path, fraction):
    # hash-based split: stable across runs and machines, no random seed to manage
    bucket = int(hashlib.md5(rel_path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def link_or_copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_source(src_root, batch_id, source, licence, holdout_fraction, patterns):
    if not src_root.exists():
        print(f"skip {source}: {src_root} not found (run scripts/download_datasets.py)")
        return
    files = sorted(p for pat in patterns for p in src_root.rglob(pat))
    landing = config.LANDING_DIR / "reference" / batch_id
    holdout = config.HOLDOUT_DIR / source
    n_ref = n_hold = 0
    for f in files:
        rel = f.relative_to(src_root).as_posix()
        if is_holdout(rel, holdout_fraction):
            link_or_copy(f, holdout / rel)
            n_hold += 1
        else:
            link_or_copy(f, landing / rel)
            n_ref += 1

    landing.mkdir(parents=True, exist_ok=True)
    (landing / "_batch.json").write_text(json.dumps({
        "source": source,
        "licence": licence,
        "landed_at": datetime.now(timezone.utc).isoformat(),
        "origin": str(src_root),
        "holdout_fraction": holdout_fraction,
    }, indent=2))
    print(f"{batch_id}: {n_ref} reference files, {n_hold} held out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma-subset", default="fma_small")
    ap.add_argument("--holdout", type=float, default=0.05)
    args = ap.parse_args()

    stage_source(config.RAW_DIR / args.fma_subset, f"ref_{args.fma_subset}", args.fma_subset,
                 "per-track CC licence (tracks.csv)", args.holdout, ["*.mp3"])
    stage_source(config.RAW_DIR / "LibriSpeech" / "dev-clean", "ref_librispeech_dev_clean",
                 "librispeech_dev_clean", "CC BY 4.0", args.holdout, ["*.flac"])


if __name__ == "__main__":
    main()
