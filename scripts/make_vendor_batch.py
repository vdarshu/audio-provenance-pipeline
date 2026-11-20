"""
Build a synthetic "vendor delivery" with known ground truth.

The public datasets contain no labelled plagiarism, so we create it:

  positives  reference-library tracks that have been altered the way copied
             audio usually is (re-encoded, resampled, trimmed, noise, gain,
             pitch shift) - the pipeline should FLAG these
  negatives  held-out tracks that are NOT in the reference library, lightly
             re-encoded so file format can't give the answer away - should be CLEAR
  defects    a few corrupt / silent files and an in-batch duplicate, to exercise
             the quality checks

The batch goes to data/landing/vendor/<batch_id>/ with neutral file names. The
answer key goes to data/ground_truth/<batch_id>.csv, outside the landing zone.

Tracks licensed NoDerivatives are never used as positives, since altering them
would go against the licence even for internal testing.

Requires: the reference batches to be ingested + validated first (it reads the
Bronze quality tables to pick usable tracks), and ffmpeg on PATH.

    python scripts/make_vendor_batch.py --batch-id vendor_synth_001 --positives 300 --negatives 300
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config  # noqa: E402

SR = 22050
# (name, weight). Weights roughly reflect how often each kind of alteration shows
# up in practice; pitch shift is deliberately included as a known-hard case.
TRANSFORMS = [
    ("exact_copy", 0.05),
    ("reencode_mp3_64k", 0.20),
    ("resample_8k", 0.15),
    ("trim_10s", 0.20),
    ("gain_minus_6db", 0.10),
    ("noise_snr_20db", 0.10),
    ("noise_snr_10db", 0.10),
    ("pitch_shift_1_semitone", 0.10),
]


def to_mp3(y, sr, dest, bitrate="128k"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp.name,
                    "-map_metadata", "-1", "-b:a", bitrate, str(dest)], check=True)
    Path(tmp.name).unlink(missing_ok=True)


def add_noise(y, snr_db, rng):
    signal_power = np.mean(y ** 2) + 1e-12
    noise = rng.normal(0, np.sqrt(signal_power / 10 ** (snr_db / 10)), size=y.shape)
    return (y + noise).astype(np.float32)


def apply_transform(name, src, dest_stem, rng):
    """Write the altered file and return its path."""
    if name == "exact_copy":
        dest = dest_stem.with_suffix(Path(src).suffix)
        dest.write_bytes(Path(src).read_bytes())
        return dest

    y, sr = librosa.load(src, sr=SR, mono=True, duration=30)
    dest = dest_stem.with_suffix(".mp3")
    if name == "reencode_mp3_64k":
        to_mp3(y, sr, dest, "64k")
    elif name == "resample_8k":
        dest = dest_stem.with_suffix(".wav")
        sf.write(dest, librosa.resample(y, orig_sr=sr, target_sr=8000), 8000)
    elif name == "trim_10s":
        seg = 10 * sr
        start = int(rng.integers(0, max(1, len(y) - seg)))
        to_mp3(y[start:start + seg], sr, dest)
    elif name == "gain_minus_6db":
        to_mp3(y * 0.5, sr, dest)
    elif name == "noise_snr_20db":
        to_mp3(add_noise(y, 20, rng), sr, dest)
    elif name == "noise_snr_10db":
        to_mp3(add_noise(y, 10, rng), sr, dest)
    elif name == "pitch_shift_1_semitone":
        to_mp3(librosa.effects.pitch_shift(y, sr=sr, n_steps=1), sr, dest)
    else:
        raise ValueError(name)
    return dest


def reference_pool():
    """Usable reference tracks: passed QC, licence allows derivatives."""
    frames = []
    for p in (config.BRONZE_DIR / "quality").glob("batch_id=ref_*/quality.parquet"):
        frames.append(pd.read_parquet(p, columns=["batch_id", "rel_path", "bronze_path",
                                                  "licence", "qc_status"]))
    if not frames:
        sys.exit("no reference batches in Bronze yet - ingest and validate them first")
    ref = pd.concat(frames)
    ref = ref[ref["qc_status"] == "pass"]
    return ref[~ref["licence"].fillna("").str.contains("NoDeriv", case=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--positives", type=int, default=300)
    ap.add_argument("--negatives", type=int, default=300)
    ap.add_argument("--vendor", default="synthetic_vendor")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out_dir = config.LANDING_DIR / "vendor" / args.batch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = reference_pool().sample(n=args.positives, random_state=args.seed)
    holdout_files = sorted(p for p in config.HOLDOUT_DIR.rglob("*")
                           if p.suffix.lower() in config.AUDIO_EXTENSIONS)
    negatives = random.sample(holdout_files, min(args.negatives, len(holdout_files)))

    names, weights = zip(*TRANSFORMS)
    truth, n = [], 0

    def next_stem():
        nonlocal n
        n += 1
        return out_dir / f"clip_{n:05d}_{rng.integers(0, 16**6):06x}"

    for r in ref.itertuples():
        tname = random.choices(names, weights)[0]
        try:
            dest = apply_transform(tname, r.bronze_path, next_stem(), rng)
        except Exception as exc:
            print(f"skip {r.rel_path}: {exc}")
            continue
        truth.append({"rel_path": dest.name, "label": "positive", "transform": tname,
                      "source_batch": r.batch_id, "source_rel_path": r.rel_path})

    for src in negatives:
        tname = random.choice(["reencode_mp3_128k", "trim_10s"])
        try:
            y, sr = librosa.load(src, sr=SR, mono=True, duration=30)
            dest = next_stem().with_suffix(".mp3")
            if tname == "trim_10s" and len(y) > 10 * sr:
                start = int(rng.integers(0, len(y) - 10 * sr))
                y = y[start:start + 10 * sr]
            to_mp3(y, sr, dest)
        except Exception as exc:
            print(f"skip {src}: {exc}")
            continue
        truth.append({"rel_path": dest.name, "label": "negative", "transform": tname,
                      "source_batch": None, "source_rel_path": None})

    # a handful of broken deliveries, which should end up as REJECTED_QC
    some_mp3 = [t for t in truth if t["rel_path"].endswith(".mp3")]
    for i in range(3):
        src = out_dir / some_mp3[i]["rel_path"]
        dest = next_stem().with_suffix(".mp3")
        dest.write_bytes(src.read_bytes()[:2048])          # truncated download
        truth.append({"rel_path": dest.name, "label": "qc_defect", "transform": "truncated"})
    for _ in range(2):
        dest = next_stem().with_suffix(".wav")
        sf.write(dest, np.zeros(SR * 10, dtype=np.float32), SR)
        truth.append({"rel_path": dest.name, "label": "qc_defect", "transform": "silent"})
    dup_src = out_dir / some_mp3[5]["rel_path"]
    dest = next_stem().with_suffix(".mp3")
    dest.write_bytes(dup_src.read_bytes())
    truth.append({"rel_path": dest.name, "label": "qc_defect", "transform": "duplicate_in_batch"})

    (out_dir / "_batch.json").write_text(json.dumps({
        "source": args.vendor,
        "licence": "vendor-asserted: commercial use permitted",   # what a vendor would claim
        "landed_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
    }, indent=2))

    config.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    gt = pd.DataFrame(truth)
    gt.to_csv(config.GROUND_TRUTH_DIR / f"{args.batch_id}.csv", index=False)
    print(gt.groupby(["label", "transform"]).size().to_string())
    print(f"\n{len(gt)} files written to {out_dir}")


if __name__ == "__main__":
    main()
