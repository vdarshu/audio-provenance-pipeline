# 2. Data sourcing and provenance

## Datasets

| | Free Music Archive (FMA) | LibriSpeech ASR corpus |
|---|---|---|
| Role in the project | music part of the reference library | speech part of the reference library |
| Subset used | Mid-submission: a **genre-balanced sample of 750 tracks** (~94 per genre, 30 s clips) from `fma_small`. The full `fma_small` (8,000 tracks) and `fma_medium` (25,000) are supported by the same scripts for the final scale test | `dev-clean`: 2,703 utterances, 40 speakers |
| Format | MP3 (mostly 44.1 kHz stereo) + `fma_metadata/tracks.csv` | FLAC, 16 kHz mono + speaker/chapter text files |
| Source | https://github.com/mdeff/fma (archives on os.unil.cloud.switch.ch) | https://www.openslr.org/12/ |
| Licence | Metadata CC BY 4.0. **Each track has its own Creative Commons licence**, recorded in `tracks.csv` (`track.license`). Mix of BY, BY-NC, BY-NC-SA, BY-NC-ND and others | CC BY 4.0 for the whole corpus |
| Citation | Defferrard et al., "FMA: A Dataset for Music Analysis", ISMIR 2017 | Panayotov et al., "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015 |

Sizes and file counts above are the published figures. The actual numbers from the download are in `data/raw/download_manifest.json` and in the Bronze quality reports (`lake/bronze/quality/batch_id=*/quality_report.json`).

## How the data was obtained

For the mid-submission the working set is kept to about 1 GB (750 FMA tracks plus LibriSpeech dev-clean, ~1.05 GB). It is the same dataset as in the proposal at a smaller scale, which is enough to build and check every pipeline stage. The full-scale run is planned for the final submission.

`fma_small` is only published as one 7.2 GB zip. In sample mode (`--sample 750`) the script reads the zip's directory and only the needed files over HTTP range requests, so the rest of the archive is never downloaded:

- tracks are chosen from `tracks.csv` where `set.subset == small`, an equal number per `genre_top`, with a fixed random seed (42), so the sample is reproducible
- zip stores a CRC-32 per file, and it is checked on extraction
- every sampled file is listed with its CRC-32 and SHA-256 in `data/raw/fma_small_sample_files.csv`
- the whole-archive SHA-1 can't be verified without downloading the full archive, so the manifest records the published value only

LibriSpeech dev-clean is small, so its full archive is downloaded and checked as described below.

For full archives, `scripts/download_datasets.py` downloads them over HTTPS from the official links above, computes a checksum for each archive, compares it to the published value and writes a provenance record to `data/raw/download_manifest.json`:

```json
"fma_small": {
  "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
  "file": "fma_small.zip",
  "bytes": ...,
  "sha1": "...",
  "checksum_matches_published": true,
  "licence": "per-track Creative Commons licences (see tracks.csv)",
  "downloaded_at": "..."
}
```

The raw archives are kept untouched in `data/raw/`. Everything downstream can be rebuilt from them.

## Licence handling in the pipeline

- Licence is carried from the source into every layer: Bronze manifest -> `ref_track.licence` -> flagged-files report. A FLAGGED file shows the licence of the work it copies.
- Reference files with no known licence fail QC (`missing_licence`) and do not enter the library.
- Tracks licensed **NoDerivatives** are never altered to build synthetic test copies (`scripts/make_vendor_batch.py` filters them out). They stay in the reference library, since matching against them doesn't modify them.
- The project is for non-commercial education and research, which is compatible with the NC licences in FMA. Neither the audio nor the synthetic copies are redistributed outside the submission.

## PII and IP check

- FMA: music released by the artists under CC licences; metadata holds artist and track names only.
- LibriSpeech: volunteers reading public-domain books for LibriVox, released publicly. `SPEAKERS.TXT` lists the reader name each volunteer published under on LibriVox. The pipeline does not load that file; speakers appear only as their numeric id (`speaker 1272`). No contact details or other personal data.
- No company-confidential data is used anywhere.

## Synthetic vendor batches (ground truth)

Neither dataset contains labelled plagiarism, so evaluation batches are generated:

1. `scripts/prepare_landing.py` holds out 5% of each dataset, chosen by a hash of the file path so the split is the same on every machine. Held-out files never enter the reference library.
2. `scripts/make_vendor_batch.py` builds a delivery from:
   - **positives**: reference tracks altered by one of: exact copy, MP3 64 kbps re-encode, resample to 8 kHz, random 10 s excerpt, -6 dB gain, white noise at 20 dB or 10 dB SNR, +1 semitone pitch shift
   - **negatives**: held-out tracks, re-encoded or trimmed so the file format doesn't give the answer away
   - **defects**: truncated files, silent files and an in-batch duplicate, to test the quality checks
3. File names are neutral (`clip_00042_3fa1c2.mp3`) and all tags are stripped. The answer key is written to `data/ground_truth/<batch_id>.csv`, outside the landing zone, because a real vendor would not send one.

These batches are clearly labelled `"synthetic": true` in their `_batch.json` and are only used to measure the pipeline.
