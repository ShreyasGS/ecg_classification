#!/usr/bin/env python3
"""
explore_data.py

Dataset exploration for ECG time series classification
- Summarize class distribution, lengths, and statistics
- Visualize representative ECGs per class
- Stratified train/validation split and justification
"""

import os
import struct
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from scipy.stats import skew, kurtosis
from scipy.signal import welch

# -----------------------------------------------------------------------------#
# Data loading (assume functions are present or import from data_loading.py)
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
    if not (os.path.exists(train_zip) and os.path.exists(test_zip) and os.path.exists(labels_csv)):
        raise FileNotFoundError("Missing training or test files in data_dir!")
    X_train = read_zip_binary(train_zip)
    y_train = pd.read_csv(labels_csv, header=None, names=["label"], dtype=int)
    X_test = read_zip_binary(test_zip)
    return X_train, y_train, X_test

# -----------------------------------------------------------------------------#
# Core statistics and plotting
# -----------------------------------------------------------------------------#
def print_class_distribution(y: pd.DataFrame):
    print("\nClass Distribution:")
    print(y["label"].value_counts().sort_index())
    print("\nClass Proportions:")
    print(y["label"].value_counts(normalize=True).sort_index())

def print_length_stats(X: List[List[int]]):
    lengths = [len(ts) for ts in X]
    print("\nLength Statistics:")
    print(pd.Series(lengths).describe(percentiles=[0.25, 0.5, 0.75]))

def print_overall_stats(X: List[List[int]]):
    lengths = [len(ts) for ts in X]
    values = np.concatenate(X)
    stats = {
        "num_samples": len(X),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "mean_value": np.mean(values),
        "std_value": np.std(values),
        "min_value": np.min(values),
        "max_value": np.max(values),
        "median_value": np.median(values),
        "q1_value": np.percentile(values, 25),
        "q3_value": np.percentile(values, 75),
    }
    print("\nOverall Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

def plot_ecg_examples(X: List[List[int]], y: pd.DataFrame, n_per_class=1, fs=300):
    """Plots representative examples for each class."""
    unique_labels = sorted(y["label"].unique())
    plt.figure(figsize=(14, 2.5 * len(unique_labels)))
    for i, cls in enumerate(unique_labels):
        idxs = np.where(y["label"].values == cls)[0]
        for j in range(n_per_class):
            plt.subplot(len(unique_labels), n_per_class, i * n_per_class + j + 1)
            plt.plot(X[idxs[j]], lw=1)
            plt.title(f"Class {cls} example {j+1}")
            plt.xlabel("Sample #")
            plt.ylabel("Amplitude")
            plt.tight_layout()
    plt.suptitle("Representative ECG examples per class", fontsize=14, y=1.02)
    plt.show()

def plot_length_vs_class(X: List[List[int]], y: pd.DataFrame):
    lengths = [len(ts) for ts in X]
    sns.violinplot(x=y['label'], y=lengths)
    plt.title("ECG Sequence Length by Class")
    plt.xlabel("Class")
    plt.ylabel("Length")
    plt.show()

# -----------------------------------------------------------------------------#
# Stratified train/validation split
# -----------------------------------------------------------------------------#
def stratified_split(X, y, val_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        stratify=y["label"],
        random_state=seed
    )
    return train_idx, val_idx

def print_split_justification(y, train_idx, val_idx):
    print("\nStratified Split - Class Proportions")
    print("Train:")
    print(y['label'].iloc[train_idx].value_counts(normalize=True).sort_index())
    print("Validation:")
    print(y['label'].iloc[val_idx].value_counts(normalize=True).sort_index())
    print(
        "\nJustification:\n"
        "A stratified 80/20 split preserves the proportion of each class in both the training and validation sets. "
        "This ensures that model performance metrics are not biased by class imbalance and that all classes are well represented in validation."
    )

# -----------------------------------------------------------------------------#
# Main logic
# -----------------------------------------------------------------------------#
def main():
    X_train, y_train, X_test = load_dataset("data")

    print_class_distribution(y_train)
    print_length_stats(X_train)
    print_overall_stats(X_train)
    
    print("\n=== Per-Class Mean/Std of Length and Amplitude ===")
    for cls in sorted(y_train["label"].unique()):
        idxs = y_train.index[y_train["label"] == cls].tolist()
        seqs = [X_train[i] for i in idxs]
        lens = [len(s) for s in seqs]
        vals = np.concatenate(seqs)
        print(f"Class {cls}: n={len(seqs)}, mean_len={np.mean(lens):.1f}±{np.std(lens):.1f}, "
              f"mean_val={np.mean(vals):.1f}±{np.std(vals):.1f}")

    # Visualize representative signals
    plot_ecg_examples(X_train, y_train, n_per_class=1)
    plot_length_vs_class(X_train, y_train)

    # Stratified split and report
    train_idx, val_idx = stratified_split(X_train, y_train, val_size=0.2, seed=42)
    print_split_justification(y_train, train_idx, val_idx)
    # Optionally: save idx or splits
    # np.savez("train_val_indices.npz", train_idx=train_idx, val_idx=val_idx)

if __name__ == "__main__":
    main()
