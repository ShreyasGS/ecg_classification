"""
data_loading.py
===============

Helper functions for loading the AMLS ECG dataset, plus feature-engineering helpers.

Directory layout expected
-------------------------
data/
│   X_train.zip      (contains *one* .bin file with all training signals)
│   X_test.zip       (contains *one* .bin file with all test signals)
│   y_train.csv      (one-column CSV with labels, no header)
└── ...

Usage
-----
from data_loading import load_dataset
X_train, y_train, X_test = load_dataset("data")

# Or run directly:
# python data_loading.py
"""

from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch


# -----------------------------------------------------------------------------#
# Low-level binary reader
# -----------------------------------------------------------------------------#

def _read_binary_from_fileobj(dst: List[List[int]], fileobj) -> None:
    """Append ECG time-series arrays to dst while reading from fileobj."""
    while True:
        length_bytes = fileobj.read(4)
        if not length_bytes:
            break
        length = struct.unpack("i", length_bytes)[0]
        signal = struct.unpack(f"{length}h", fileobj.read(length * 2))
        dst.append(list(signal))


# -----------------------------------------------------------------------------#
# Public helpers
# -----------------------------------------------------------------------------#

def read_zip_binary(zip_path: Path | str) -> List[List[int]]:
    """
    Read ECG signals directly from a ZIP containing exactly one .bin file.
    Returns a ragged list where each element is one ECG time series.
    """
    zip_path = Path(zip_path)
    signals: List[List[int]] = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # find .bin
        bin_name = next((f for f in zf.namelist() if f.endswith('.bin')), None)
        if not bin_name:
            raise FileNotFoundError(f"No .bin file inside {zip_path}")
        with zf.open(bin_name, 'r') as f:
            _read_binary_from_fileobj(signals, f)
    return signals


def load_labels(csv_path: Path | str) -> pd.Series:
    """
    Load the single-column y_train.csv and return a Pandas Series 'label'.
    """
    labels = pd.read_csv(csv_path, header=None, names=['label'], dtype=int)
    return labels['label']


def load_dataset(data_dir: Path | str = 'data') -> Tuple[List[List[int]], pd.Series, List[List[int]]]:
    """
    Load training signals, labels, and test signals in one shot.

    Returns:
      - X_train: List of ECG lists
      - y_train: Pandas Series of labels
      - X_test:  List of ECG lists
    """
    data_dir = Path(data_dir)
    paths = {
        'train_zip': data_dir / 'X_train.zip',
        'test_zip':  data_dir / 'X_test.zip',
        'labels':    data_dir / 'y_train.csv'
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    print('🔹 Loading training data …')
    X_train = read_zip_binary(paths['train_zip'])
    print('🔹 Loading labels …')
    y_train = load_labels(paths['labels'])
    print('🔹 Loading test data …')
    X_test = read_zip_binary(paths['test_zip'])
    print(f"✅ Loaded {len(X_train)} train samples, {len(X_test)} test samples.")
    return X_train, y_train, X_test


# -----------------------------------------------------------------------------#
# Feature engineering helpers
# -----------------------------------------------------------------------------#

def summarize_signal(sig: List[int]) -> dict[str, float]:
    """
    Compute time-domain summaries for one ECG signal.
    Returns keys: ptp, mad, skew, kurt, zcr, energy, entropy.
    """
    arr = np.array(sig)
    ptp = float(arr.max() - arr.min())
    mad = float(np.mean(np.abs(arr - arr.mean())))
    sk = float(skew(arr))
    kt = float(kurtosis(arr))
    zcr = float(np.mean(np.diff(np.sign(arr)) != 0))
    energy = float(np.sum(arr**2))
    hist = np.histogram(arr, bins=50)[0] + 1
    probs = hist / hist.sum()
    ent = float(-np.sum(probs * np.log2(probs)))
    return {
        'ptp': ptp,
        'mad': mad,
        'skew': sk,
        'kurt': kt,
        'zcr': zcr,
        'energy': energy,
        'entropy': ent
    }


def noise_ratio(sig: List[int], fs: int = 300) -> float:
    """
    Compute the ratio of high-frequency (>40Hz) power to total power.
    """
    f, Pxx = welch(sig, fs=fs, nperseg=1024)
    hf = np.trapz(Pxx[f > 40], f[f > 40])
    tot = np.trapz(Pxx, f)
    return float(hf / tot)


# -----------------------------------------------------------------------------#
# Quick sanity check when executed directly
# -----------------------------------------------------------------------------#
if __name__ == '__main__':
    X_train, y_train, X_test = load_dataset('data')
    print('\nFirst sample length:', len(X_train[0]))
    print('\nLabel distribution:')
    print(y_train.value_counts().sort_index())
