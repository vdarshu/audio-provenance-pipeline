"""Quick sanity checks for the fingerprinting code. Run with: python -m pytest tests"""
import numpy as np

from pipeline import config
from pipeline.fingerprint import embedding, find_peaks, landmark_hashes, EMBEDDING_DIM

SR = config.FP_SAMPLE_RATE


def tone_sequence(seed, seconds=10):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * 0.25)) / SR
    notes = [
        (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) + 0.3 * np.sin(6 * np.pi * f * t))
        * np.exp(-4 * t)
        for f in rng.uniform(150, 1200, seconds * 4)
    ]
    return np.concatenate(notes).astype(np.float32)


def aligned_votes(a, b):
    """Best count of hash hits sharing one time offset (what the gold layer does in SQL)."""
    ha, ta = landmark_hashes(find_peaks(a))
    hb, tb = landmark_hashes(find_peaks(b))
    index = {}
    for h, t in zip(ha, ta):
        index.setdefault(h, []).append(t)
    deltas = {}
    for h, t in zip(hb, tb):
        for t_ref in index.get(h, []):
            deltas[t_ref - t] = deltas.get(t_ref - t, 0) + 1
    return max(deltas.values(), default=0)


def test_embedding_shape_and_norm():
    vec = embedding(tone_sequence(1))
    assert vec.shape == (EMBEDDING_DIM,)
    assert abs(np.linalg.norm(vec) - 1) < 1e-4


def test_hashes_fit_in_postgres_int():
    hashes, _ = landmark_hashes(find_peaks(tone_sequence(2)))
    assert len(hashes) > 0
    assert hashes.max() < 2**31


def test_trimmed_noisy_copy_matches_but_other_audio_does_not():
    original = tone_sequence(3)
    rng = np.random.default_rng(0)
    excerpt = original[SR * 2: SR * 9]
    noise_std = np.sqrt(np.mean(excerpt ** 2) / 10 ** (20 / 10))     # 20 dB SNR
    noisy = excerpt + rng.normal(0, noise_std, excerpt.shape).astype(np.float32)
    unrelated = tone_sequence(4)[SR * 2: SR * 9]

    assert aligned_votes(original, noisy) >= config.MATCH_MIN_VOTES
    assert aligned_votes(original, unrelated) < config.MATCH_REVIEW_VOTES
