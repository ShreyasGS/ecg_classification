#!/usr/bin/env python3
"""
explore_data.py

1. Load X_train, y_train (as DataFrame), X_test
2. Compute overall ECG stats
3. Compute per-class ECG stats
4. Compute advanced ECG-specific features and visualizations
"""

import os
import struct
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.signal import welch, spectrogram
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: for HRV and beat detection
try:
    from biosppy.signals import ecg
except ImportError:
    ecg = None  # Requires `pip install biosppy`

# Optional: for DTW clustering
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    fastdtw = None  # Requires `pip install fastdtw`


# -----------------------------------------------------------------------------#
# Data loading
# -----------------------------------------------------------------------------#

def read_zip_binary(zip_path: str) -> List[List[int]]:
    signals: List[List[int]] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        bin_name = next(f for f in zf.namelist() if f.endswith(".bin"))
        with zf.open(bin_name, "r") as f:
            while True:
                size_bytes = f.read(4)
                if not size_bytes:
                    break
                n = struct.unpack("i", size_bytes)[0]
                vals = struct.unpack(f"{n}h", f.read(n * 2))
                signals.append(list(vals))
    return signals


def load_dataset(data_dir: str = "data") -> Tuple[List[List[int]], pd.DataFrame, List[List[int]]]:
    train_zip = os.path.join(data_dir, "X_train.zip")
    test_zip = os.path.join(data_dir, "X_test.zip")
    labels_csv = os.path.join(data_dir, "y_train.csv")
    missing = [p for p in (train_zip, test_zip, labels_csv) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing file(s): {missing}")

    print("🔹 Loading training data …")
    X_train = read_zip_binary(train_zip)
    print("🔹 Loading labels …")
    y_train_df = pd.read_csv(labels_csv, header=None, names=["label"], dtype=int)
    print("🔹 Loading test data …")
    X_test = read_zip_binary(test_zip)

    print(f"✅ Loaded {len(X_train):,} train, {len(X_test):,} test samples.")
    return X_train, y_train_df, X_test


# -----------------------------------------------------------------------------#
# Basic statistics
# -----------------------------------------------------------------------------#

def compute_overall_statistics(X: List[List[int]]) -> dict:
    lengths = [len(ts) for ts in X]
    values = np.concatenate(X)
    return {
        "num_samples": len(X),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "mean_value": float(np.mean(values)),
        "std_value": float(np.std(values)),
        "min_value": int(np.min(values)),
        "max_value": int(np.max(values)),
        "median_value": float(np.median(values)),
        "q1_value": float(np.percentile(values, 25)),
        "q3_value": float(np.percentile(values, 75)),
    }


def compute_class_statistics(X: List[List[int]], y_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for cls in sorted(y_df["label"].unique()):
        idxs = y_df.index[y_df["label"] == cls].tolist()
        seqs = [X[i] for i in idxs]
        lengths = [len(s) for s in seqs]
        vals = np.concatenate(seqs)
        records.append({
            "label": cls,
            "count": len(seqs),
            "mean_length": float(np.mean(lengths)),
            "mean_value": float(np.mean(vals)),
            "std_value": float(np.std(vals)),
        })
    return pd.DataFrame.from_records(records).set_index("label")


# -----------------------------------------------------------------------------#
# Time-domain features
# -----------------------------------------------------------------------------#

def summarize_signal(sig: List[int]) -> dict:
    arr = np.array(sig)
    return {
        "ptp": float(arr.max() - arr.min()),
        "mad": float(np.mean(np.abs(arr - arr.mean()))),
        "skew": float(st.skew(arr)),
        "kurt": float(st.kurtosis(arr)),
        "zcr": float(np.mean(np.diff(np.sign(arr)) != 0)),
        "energy": float(np.sum(arr**2)),
        "entropy": float(st.entropy(np.histogram(arr, bins=50)[0] + 1)),
    }


def compute_time_domain_features(X: List[List[int]], y_df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for sig, lbl in zip(X, y_df['label']):
        d = summarize_signal(sig)
        d['label'] = lbl
        feats.append(d)
    return pd.DataFrame(feats)


# -----------------------------------------------------------------------------#
# Heart-rate & variability (requires BioSPPy)
# -----------------------------------------------------------------------------#

def compute_hrv_features(X: List[List[int]], y_df: pd.DataFrame, fs: int = 300) -> pd.DataFrame:
    if ecg is None:
        raise ImportError("biosppy not installed: pip install biosppy")
    records = []
    for sig, lbl in zip(X, y_df['label']):
        out = ecg.ecg(signal=np.array(sig), sampling_rate=fs, show=False)
        rri = np.diff(out['rpeaks']) / fs * 1000
        records.append({
            'label': lbl,
            'hr_mean': float(60000 / np.mean(rri)),
            'sdnn': float(np.std(rri)),
            'rmssd': float(np.sqrt(np.mean(np.diff(rri)**2)))
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------#
# Frequency-domain features
# -----------------------------------------------------------------------------#

def compute_frequency_features(X: List[List[int]], y_df: pd.DataFrame, fs: int = 300) -> pd.DataFrame:
    records = []
    for sig, lbl in zip(X, y_df['label']):
        f, Pxx = welch(sig, fs=fs, nperseg=1024)
        # capture e.g. band powers
        records.append({
            'label': lbl,
            'power_low': float(np.trapz(Pxx[(f>=0.5)&(f<4)] , f[(f>=0.5)&(f<4)])),
            'power_high': float(np.trapz(Pxx[(f>=20)&(f<50)], f[(f>=20)&(f<50)]))
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------#
# Noise metric (high-frequency energy ratio)
# -----------------------------------------------------------------------------#

def compute_noise_metric(X: List[List[int]], y_df: pd.DataFrame, fs: int = 300) -> pd.DataFrame:
    recs = []
    for sig, lbl in zip(X, y_df['label']):
        f, Pxx = welch(sig, fs=fs, nperseg=1024)
        hf = np.trapz(Pxx[f > 40], f[f > 40])
        tot = np.trapz(Pxx, f)
        recs.append({'label': lbl, 'noise_ratio': float(hf/tot)})
    return pd.DataFrame(recs)


# -----------------------------------------------------------------------------#
# DTW clustering (requires fastdtw)
# -----------------------------------------------------------------------------#

def dtw_distance(a: List[int], b: List[int]) -> float:
    if fastdtw is None:
        raise ImportError("fastdtw not installed: pip install fastdtw")
    dist, _ = fastdtw(a, b, dist=euclidean)
    return dist


def compute_dtw_clusters(X: List[List[int]], n_samples: int = 50) -> np.ndarray:
    # Sample subset for distance matrix
    idxs = np.random.choice(len(X), n_samples, replace=False)
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            d = dtw_distance(X[idxs[i]], X[idxs[j]])
            D[i,j] = D[j,i] = d
    return D


# -----------------------------------------------------------------------------#
# Correlation of length and class
# -----------------------------------------------------------------------------#

def plot_length_vs_class(X: List[List[int]], y_df: pd.DataFrame) -> None:
    lengths = [len(ts) for ts in X]
    sns.violinplot(x=y_df['label'], y=lengths)
    plt.title("Sequence Length by Class")
    plt.show()


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def main():
    X_train, y_train_df, X_test = load_dataset("data")

    print("\nComputing overall statistics...")
    overall = compute_overall_statistics(X_train)
    for k, v in overall.items():
        print(f"  {k}: {v}")

    print("\nComputing class statistics...")
    class_stats = compute_class_statistics(X_train, y_train_df)
    print(class_stats)

    # Advanced features
    print("\nComputing time-domain features...")
    df_td = compute_time_domain_features(X_train, y_train_df)
    print(df_td.groupby('label').mean())

    if ecg is not None:
        print("\nComputing HRV features...")
        df_hrv = compute_hrv_features(X_train, y_train_df)
        print(df_hrv.groupby('label').mean())

    print("\nComputing frequency-domain features...")
    df_fd = compute_frequency_features(X_train, y_train_df)
    print(df_fd.groupby('label').mean())

    print("\nComputing noise metric...")
    df_noise = compute_noise_metric(X_train, y_train_df)
    print(df_noise.groupby('label').mean())

    print("\nPlotting length vs class...")
    plot_length_vs_class(X_train, y_train_df)

    if fastdtw is not None:
        print("\nComputing DTW distance matrix for sample subset...")
        D = compute_dtw_clusters(X_train, n_samples=20)
        print("Distance matrix shape:", D.shape)


if __name__ == "__main__":
    main()
