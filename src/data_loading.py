"""
data_loading.py
===============

Helper functions for loading the AMLS ECG dataset.

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
python data_loading.py
"""

from __future__ import annotations

import os
import struct
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# -----------------------------------------------------------------------------#
# Low-level binary reader
# -----------------------------------------------------------------------------#
def _read_binary_from_fileobj(dst: List[List[int]], fileobj) -> None:
    """Append ECG time-series arrays to *dst* while reading from *fileobj*."""
    while True:
        length_bytes = fileobj.read(4)
        if not length_bytes:                            # EOF
            break
        length = struct.unpack("i", length_bytes)[0]    # 32-bit length
        signal = struct.unpack(f"{length}h", fileobj.read(length * 2))
        dst.append(list(signal))


# -----------------------------------------------------------------------------#
# Public helpers
# -----------------------------------------------------------------------------#
def read_zip_binary(zip_path: Path | str) -> List[List[int]]:
    """
    Read ECG signals *directly* from a ZIP that contains **exactly one** ``.bin`` file.

    Returns
    -------
    List[List[int]]
        A ragged list; each inner list is one ECG time series.
    """
    zip_path = Path(zip_path)
    signals: List[List[int]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Auto-detect the .bin file inside (first match)
        try:
            bin_name = next(name for name in zf.namelist() if name.endswith(".bin"))
        except StopIteration as exc:
            raise FileNotFoundError(f"No .bin file found inside {zip_path}") from exc

        with zf.open(bin_name, "r") as f:
            _read_binary_from_fileobj(signals, f)

    return signals


def load_labels(csv_path: Path | str) -> pd.Series:
    """
    Load the *single-column* ``y_train.csv`` and return it as a Pandas Series.

    All values are cast to ``int``; the Series is named ``label``.
    """
    labels = pd.read_csv(csv_path, header=None, names=["label"], dtype=int)
    return labels["label"]


def load_dataset(data_dir: Path | str = "data") -> Tuple[List[List[int]],
                                                         pd.Series,
                                                         List[List[int]]]:
    """
    Load training signals, labels and test signals in one shot.

    Parameters
    ----------
    data_dir : str or Path
        Folder containing ``X_train.zip``, ``X_test.zip`` and ``y_train.csv``.

    Returns
    -------
    (X_train, y_train, X_test)
        * X_train : List of ECG signals (ragged)
        * y_train : Pandas Series of integer labels
        * X_test  : List of ECG signals (ragged)
    """
    data_dir = Path(data_dir)

    paths = {
        "train_zip": data_dir / "X_train.zip",
        "test_zip":  data_dir / "X_test.zip",
        "labels":    data_dir / "y_train.csv",
    }

    # Basic existence check
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing file(s) in {data_dir}: {', '.join(missing)}")

    print("🔹 Loading training data …")
    X_train = read_zip_binary(paths["train_zip"])

    print("🔹 Loading labels …")
    y_train = load_labels(paths["labels"])

    print("🔹 Loading test data …")
    X_test = read_zip_binary(paths["test_zip"])

    print(f"✅  Loaded {len(X_train):,} train samples, {len(X_test):,} test samples.")
    return X_train, y_train, X_test


# -----------------------------------------------------------------------------#
# Quick sanity check when executed directly
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    X_train, y_train, X_test = load_dataset("data")

    print("\nFirst sample length:", len(X_train[0]))
    print("\nLabel distribution:")
    print(y_train.value_counts().sort_index())
