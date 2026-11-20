"""
Per-file metadata for the public reference sources.

FMA ships a tracks.csv with a two-row header (column groups like
('track', 'title')). LibriSpeech has no per-file licence (the whole corpus is
CC BY 4.0) but speaker and chapter ids can be read from the path.
"""
from pathlib import Path

import pandas as pd

from pipeline import config

FMA_TRACKS_CSV = config.RAW_DIR / "fma_metadata" / "tracks.csv"


def load_fma_tracks(path=FMA_TRACKS_CSV):
    if not Path(path).exists():
        return None
    tracks = pd.read_csv(path, index_col=0, header=[0, 1], low_memory=False)
    out = pd.DataFrame({
        "title": tracks[("track", "title")],
        "artist": tracks[("artist", "name")],
        "licence": tracks[("track", "license")],
    })
    out.index = out.index.astype(int)
    return out


def enrich(df, source, default_licence=None):
    """
    Add title / artist / licence columns to a Bronze manifest DataFrame.
    Falls back to the batch-level licence when nothing better is known.
    """
    df = df.copy()
    df["title"] = None
    df["artist"] = None
    df["licence"] = default_licence

    if source.startswith("fma"):
        tracks = load_fma_tracks()
        # FMA files are named by track id, e.g. 000/000002.mp3 -> 2
        track_ids = pd.to_numeric(
            df["rel_path"].str.extract(r"(\d+)\.\w+$")[0], errors="coerce"
        )
        df["source_track_id"] = track_ids
        if tracks is not None:
            joined = tracks.reindex(track_ids.astype("Int64"))
            df["title"] = joined["title"].to_numpy()
            df["artist"] = joined["artist"].to_numpy()
            df["licence"] = joined["licence"].fillna(default_licence or "unknown").to_numpy()

    elif source.startswith("librispeech"):
        # <speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
        parts = df["rel_path"].str.split("/", expand=True)
        df["artist"] = "speaker " + parts[0]
        df["title"] = df["rel_path"].str.extract(r"([^/]+)\.\w+$")[0]
        df["licence"] = default_licence or "CC BY 4.0"

    return df
