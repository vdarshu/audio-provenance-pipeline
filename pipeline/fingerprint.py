"""
Audio fingerprinting.

Two representations are produced for every file:

1. A compact 72-d embedding (MFCC / chroma / spectral contrast statistics).
   Cheap to compare and good for "these sound alike" candidate retrieval with
   pgvector, but on its own it cannot prove that one file is a copy of another -
   two songs in the same genre can sit close together.

2. Landmark hashes, the classic peak-pair approach from Wang (2003), the
   paper behind Shazam. Spectrogram peaks survive re-encoding, resampling, gain
   changes and moderate noise, and because each hash carries a time offset,
   a copy produces many hits that line up at the same offset difference. That
   alignment is what we use as evidence of copying, and it also works when the
   incoming clip is only a trimmed piece of the original.
"""
import numpy as np
import librosa
from scipy.ndimage import maximum_filter

from pipeline import config

EMBEDDING_DIM = 72


def load_audio(path, max_seconds=config.FP_MAX_SECONDS):
    """Decode to mono float32 at the fingerprint sample rate."""
    y, _ = librosa.load(path, sr=config.FP_SAMPLE_RATE, mono=True, duration=max_seconds)
    return y


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embedding(y, sr=config.FP_SAMPLE_RATE):
    """
    Summary statistics of three feature families. Each block is L2-normalised on
    its own before concatenation, otherwise the MFCCs (large values) would drown
    out chroma and contrast in the cosine distance.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=config.FP_N_FFT,
                                hop_length=config.FP_HOP)[1:]   # drop c0 = loudness
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=config.FP_N_FFT,
                                         hop_length=config.FP_HOP)
    # 4 octave bands from 200 Hz stay under the 5.5 kHz Nyquist limit at 11 kHz
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=config.FP_N_FFT,
                                                 hop_length=config.FP_HOP,
                                                 fmin=200.0, n_bands=4)
    blocks = []
    for feat in (mfcc, chroma, contrast):
        blocks.append(_l2(np.concatenate([feat.mean(axis=1), feat.std(axis=1)])))
    vec = _l2(np.concatenate(blocks)).astype(np.float32)
    assert vec.shape[0] == EMBEDDING_DIM
    return vec


def find_peaks(y):
    """
    Return spectrogram peaks as (frame, freq_bin) pairs sorted by time.

    A point counts as a peak if it is the maximum in its neighbourhood and is
    reasonably loud. Density is capped per second so a dense rock track doesn't
    produce 10x more hashes than a speech file.
    """
    spec = np.abs(librosa.stft(y, n_fft=config.FP_N_FFT, hop_length=config.FP_HOP))
    if spec.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)

    local_max = maximum_filter(log_spec, size=config.FP_PEAK_NEIGHBOURHOOD) == log_spec
    loud_enough = log_spec > -50   # dB relative to the loudest bin in the file
    freq_idx, frame_idx = np.nonzero(local_max & loud_enough)
    if len(frame_idx) == 0:
        return np.empty((0, 2), dtype=np.int32)

    # keep the strongest peaks per one-second window
    frames_per_sec = config.FP_SAMPLE_RATE / config.FP_HOP
    window = (frame_idx / frames_per_sec).astype(int)
    strength = log_spec[freq_idx, frame_idx]
    keep = np.zeros(len(frame_idx), dtype=bool)
    for w in np.unique(window):
        idx = np.nonzero(window == w)[0]
        top = idx[np.argsort(strength[idx])[::-1][: config.FP_PEAKS_PER_SECOND]]
        keep[top] = True

    peaks = np.stack([frame_idx[keep], freq_idx[keep]], axis=1).astype(np.int32)
    return peaks[np.lexsort((peaks[:, 1], peaks[:, 0]))]


def landmark_hashes(peaks):
    """
    Pair every anchor peak with the next few peaks inside a short target zone.
    Hash layout (28 bits, fits a Postgres INT):  f1 (11 bits) | f2 (11 bits) | dt (6 bits).
    Returns two int arrays: hashes and the anchor frame for each hash.
    """
    hashes, offsets = [], []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        paired = 0
        for j in range(i + 1, n):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt == 0:
                continue
            if dt > config.FP_MAX_DT:
                break
            hashes.append((int(f1) << 17) | (int(f2) << 6) | int(dt & 0x3F))
            offsets.append(int(t1))
            paired += 1
            if paired >= config.FP_FAN_OUT:
                break
    return np.asarray(hashes, dtype=np.int64), np.asarray(offsets, dtype=np.int32)


def fingerprint_file(path):
    """Everything the Silver layer stores for one file."""
    y = load_audio(path)
    peaks = find_peaks(y)
    hashes, offsets = landmark_hashes(peaks)
    return {
        "duration_s": float(len(y) / config.FP_SAMPLE_RATE),
        "embedding": embedding(y),
        "hashes": hashes,
        "offsets": offsets,
    }


def frames_to_seconds(frames):
    return frames * config.FP_HOP / config.FP_SAMPLE_RATE
