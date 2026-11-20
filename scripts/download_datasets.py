"""
Download the public reference datasets into data/raw/ and record provenance.

Sources
  FMA (Free Music Archive)   https://github.com/mdeff/fma
  LibriSpeech ASR corpus     https://www.openslr.org/12/

Each archive's checksum is computed after download and written, together with
the URL and download time, to data/raw/download_manifest.json. That file is
the evidence for docs/02_data_sourcing_and_provenance.md.

Usage:
    python scripts/download_datasets.py --sample 750    # ~1 GB: 750 FMA tracks + dev-clean
    python scripts/download_datasets.py                 # full fma_small + metadata + dev-clean
    python scripts/download_datasets.py --fma medium    # bigger FMA subset (~22 GB)
    python scripts/download_datasets.py --no-extract

Sample mode
    fma_small is only published as one 7.2 GB zip. With --sample N the script
    reads just tracks.csv and N tracks straight out of the remote zips using
    HTTP range requests (remotezip), so only ~1 GB is transferred. Tracks are
    picked evenly across the 8 genres with a fixed seed, so the same sample
    comes back on every run. The archive checksum can't be checked without the
    whole file, but zip stores a CRC-32 for every member and it is verified on
    extraction; those CRCs and a sha256 per file go into
    data/raw/fma_small_sample_files.csv.
"""
import argparse
import csv
import hashlib
import json
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config  # noqa: E402

# Published checksums (FMA README / OpenSLR). A mismatch is reported but not fatal -
# the computed value is what gets recorded in the manifest either way.
DATASETS = {
    "fma_metadata": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "algo": "sha1", "expected": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
        "licence": "CC BY 4.0 (metadata); per-track licences listed in tracks.csv",
    },
    "fma_small": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "algo": "sha1", "expected": "ade154f733639d52e35e32f5593efe5be76c6d70",
        "licence": "per-track Creative Commons licences (see tracks.csv)",
    },
    "fma_medium": {
        "url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
        "algo": "sha1", "expected": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
        "licence": "per-track Creative Commons licences (see tracks.csv)",
    },
    "librispeech_dev_clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "algo": "md5", "expected": "42e2234ba48799c1f50f24a7926300a1",
        "licence": "CC BY 4.0",
    },
}


def download(url, dest):
    if dest.exists():
        print(f"  {dest.name} already downloaded")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)


def checksum(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def extract(archive, target):
    print(f"  extracting {archive.name}")
    if archive.suffix == ".zip":
        # if this fails on a huge FMA zip, extract with 7-Zip instead (known FMA issue)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")


def download_fma_sample(n_tracks, seed=42):
    """Pull tracks.csv and a genre-balanced sample of fma_small from the remote zips."""
    import pandas as pd
    from remotezip import RemoteZip   # pip install remotezip

    meta_spec, audio_spec = DATASETS["fma_metadata"], DATASETS["fma_small"]

    tracks_csv = config.RAW_DIR / "fma_metadata" / "tracks.csv"
    if not tracks_csv.exists():
        print("  reading tracks.csv from fma_metadata.zip")
        with RemoteZip(meta_spec["url"]) as z:
            z.extract("fma_metadata/tracks.csv", config.RAW_DIR)

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
    small = tracks[tracks[("set", "subset")] == "small"]
    genre = small[("track", "genre_top")]
    per_genre = max(1, n_tracks // genre.nunique())
    picked = (genre.groupby(genre, group_keys=False)
              .apply(lambda g: g.sample(n=min(per_genre, len(g)), random_state=seed))
              .index.sort_values())
    print(f"  sampling {len(picked)} tracks ({per_genre} per genre)")

    rows = []
    with RemoteZip(audio_spec["url"]) as z:
        members = {info.filename: info for info in z.infolist()}
        for i, tid in enumerate(picked, 1):
            name = f"fma_small/{tid // 1000:03d}/{tid:06d}.mp3"
            info = members.get(name)
            if info is None:
                print(f"  {name} not in archive, skipped")
                continue
            dest = config.RAW_DIR / name
            if not dest.exists():
                z.extract(name, config.RAW_DIR)      # raises if the CRC-32 doesn't match
            rows.append({"track_id": tid, "member": name, "bytes": info.file_size,
                         "zip_crc32": f"{info.CRC:08x}", "sha256": checksum(dest, "sha256"),
                         "genre": genre.get(tid)})
            if i % 50 == 0:
                print(f"\r  {i}/{len(picked)} tracks", end="", flush=True)
    print()

    listing = config.RAW_DIR / "fma_small_sample_files.csv"
    with open(listing, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "fma_metadata": {
            "url": meta_spec["url"], "member": "fma_metadata/tracks.csv",
            "sha256": checksum(tracks_csv, "sha256"), "licence": meta_spec["licence"],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
        "fma_small_sample": {
            "url": audio_spec["url"],
            "published_archive_sha1": audio_spec["expected"],
            "tracks": len(rows),
            "bytes": sum(r["bytes"] for r in rows),
            "sampling": f"{per_genre} tracks per genre_top, pandas sample random_state={seed}",
            "file_list": listing.name,
            "integrity": "zip CRC-32 verified per file on extraction; sha256 per file in file_list",
            "licence": audio_spec["licence"],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fma", choices=["small", "medium"], default="small")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--sample", type=int, metavar="N",
                    help="only fetch N fma_small tracks (genre-balanced) instead of the full zip")
    args = ap.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = config.RAW_DIR / "download_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}

    if args.sample:
        print("fma (sample)")
        manifest.update(download_fma_sample(args.sample))
        full_downloads = ["librispeech_dev_clean"]
    else:
        full_downloads = ["fma_metadata", f"fma_{args.fma}", "librispeech_dev_clean"]

    for name in full_downloads:
        spec = DATASETS[name]
        archive = config.RAW_DIR / Path(spec["url"]).name
        print(name)
        download(spec["url"], archive)
        digest = checksum(archive, spec["algo"])
        ok = digest == spec["expected"]
        if not ok:
            print(f"  WARNING: {spec['algo']} {digest} does not match published {spec['expected']}")
        manifest[name] = {
            "url": spec["url"],
            "file": archive.name,
            "bytes": archive.stat().st_size,
            spec["algo"]: digest,
            "checksum_matches_published": ok,
            "licence": spec["licence"],
            "downloaded_at": manifest.get(name, {}).get("downloaded_at")
                             or datetime.now(timezone.utc).isoformat(),
        }
        if not args.no_extract:
            extract(archive, config.RAW_DIR)

    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"provenance written to {manifest_file}")


if __name__ == "__main__":
    main()
