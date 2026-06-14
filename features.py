"""
features.py
===========
Single source of truth for audio loading + feature extraction.

The IDENTICAL code is used by every part of the project:
    - notebook.ipynb   (written to disk via %%writefile, then imported)
    - train_pipeline.py
    - predict.py
    - app.py

⚠️  If you change ANY feature parameter (sample rate, duration, feature
    set, or ordering) you MUST keep all copies identical. A scaler/model
    trained on one feature layout will silently mis-score audio extracted
    with a different layout.
"""
from __future__ import annotations
import warnings

import numpy as np
import librosa

# --- Audio configuration (identical everywhere) ---
SR: int = 16_000            # target sample rate (mono)
DURATION: float = 4.0       # seconds; clips padded/truncated to this
N_SAMPLES: int = int(SR * DURATION)   # 64_000
N_MFCC: int = 40

# Vector length returned by extract_features():
#   MFCC mean/std 40+40=80 | chroma 12+12=24 | contrast 7+7=14
#   ZCR 1+1=2 | RMS 1+1=2 | rolloff 1+1=2  -> total = 124
N_FEATURES: int = 124


def load_audio(path: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Load `path` as mono at `sr`, then pad/truncate to `duration` seconds."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        y = y[:target]
    return y


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return a fixed-length (N_FEATURES,) float32 feature vector.

    Order is fixed — do not reorder:
        [mfcc_mean(40), mfcc_std(40), chroma_mean(12), chroma_std(12),
         contrast_mean(7), contrast_std(7), zcr_mean, zcr_std,
         rms_mean, rms_std, rolloff_mean, rolloff_std]
    """
    y = np.asarray(y, dtype=np.float32)
    if not np.any(y):                      # guard fully-silent clips
        y = y + 1e-10

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zcr      = librosa.feature.zero_crossing_rate(y=y)
        rms      = librosa.feature.rms(y=y)
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)

    feats = np.concatenate([
        np.mean(mfcc,     axis=1), np.std(mfcc,     axis=1),
        np.mean(chroma,   axis=1), np.std(chroma,   axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        [np.mean(zcr)],     [np.std(zcr)],
        [np.mean(rms)],     [np.std(rms)],
        [np.mean(rolloff)], [np.std(rolloff)],
    ]).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features_from_file(path: str, sr: int = SR,
                               duration: float = DURATION) -> np.ndarray:
    """Convenience wrapper: load a file then extract its feature vector."""
    return extract_features(load_audio(path, sr=sr, duration=duration), sr=sr)


if __name__ == "__main__":
    t = np.linspace(0, 1, SR, endpoint=False)
    demo = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    v = extract_features(demo)
    print(f"feature vector length = {v.shape[0]} (expected {N_FEATURES})")
    assert v.shape[0] == N_FEATURES
    print("features.py OK")
