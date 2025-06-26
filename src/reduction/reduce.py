"""
reduce.py

ECG Data Reduction and Compression Utilities
--------------------------------------------
- Downsampling, piecewise constant/linear, wavelet, Fourier, quantization, delta, PCA.
- Coreset selection (random or kmeans).
- Custom binary serialization/compression and reader.
- Example: runs for 10%, 25%, 50% data reduction.
"""

import numpy as np
import os
import struct
import pickle
import zlib
import pywt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random
import sys
import pandas as pd

# Always add src/ to sys.path for imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REDUCED_DIR = os.path.join(PROJECT_ROOT, "reduced")
os.makedirs(REDUCED_DIR, exist_ok=True)

from reduction.metrics import compute_all_metrics


### --- Data Reduction Techniques ---

def downsample(ts, factor=2):
    return np.array(ts)[::factor]

def piecewise_constant_approximation(ts, num_segments):
    seg_len = len(ts) // num_segments
    out = np.zeros(len(ts))
    for i in range(num_segments):
        start = i * seg_len
        end = len(ts) if i == num_segments-1 else (i+1)*seg_len
        out[start:end] = np.mean(ts[start:end])
    return out

def piecewise_linear_approximation(ts, num_segments):
    seg_len = len(ts) // num_segments
    out = np.zeros(len(ts))
    for i in range(num_segments):
        start = i * seg_len
        end = len(ts) if i == num_segments-1 else (i+1)*seg_len
        x = np.arange(start, end)
        y = ts[start:end]
        if len(x) < 2:
            out[start:end] = y
        else:
            coeffs = np.polyfit(x, y, 1)
            out[start:end] = np.polyval(coeffs, x)
    return out

def wavelet_compression(ts, wavelet='db4', threshold_ratio=0.1):
    coeffs = pywt.wavedec(ts, wavelet)
    arr, slices = pywt.coeffs_to_array(coeffs)
    idx = np.argsort(np.abs(arr))
    n_keep = int(len(arr) * threshold_ratio)
    arr[idx[:-n_keep]] = 0
    coeffs_thresh = pywt.array_to_coeffs(arr, slices, output_format='wavedec')
    recon = pywt.waverec(coeffs_thresh, wavelet)
    return recon[:len(ts)]

def fourier_compression(ts, threshold_ratio=0.1):
    f = np.fft.rfft(ts)
    idx = np.argsort(np.abs(f))
    n_keep = int(len(f) * threshold_ratio)
    f[idx[:-n_keep]] = 0
    recon = np.fft.irfft(f, n=len(ts))
    return recon

def quantize(ts, num_levels=256):
    min_val, max_val = np.min(ts), np.max(ts)
    q = np.round((ts - min_val) / (max_val - min_val) * (num_levels-1))
    return q.astype(np.uint8), float(min_val), float(max_val)

def dequantize(q, min_val, max_val, num_levels=256):
    return q / (num_levels-1) * (max_val - min_val) + min_val

def delta_encoding(ts):
    ts = np.asarray(ts)
    return np.diff(ts), ts[0]

def delta_decoding(deltas, first_val):
    return np.concatenate([[first_val], np.cumsum(deltas) + first_val])

def coreset_selection(ts_list, y, ratio=0.1, method='kmeans'):
    n = max(1, int(len(ts_list) * ratio))
    if method == "random":
        idx = random.sample(range(len(ts_list)), n)
        return [ts_list[i] for i in idx], y.iloc[idx]
    # KMeans on mean, std, ptp, median
    feats = np.array([[np.mean(ts), np.std(ts), np.ptp(ts), np.median(ts)] for ts in ts_list])
    kmeans = KMeans(n_clusters=n, random_state=42).fit(feats)
    sel_idx = []
    for i in range(n):
        cluster_idx = np.where(kmeans.labels_ == i)[0]
        c = kmeans.cluster_centers_[i]
        d = np.linalg.norm(feats[cluster_idx] - c, axis=1)
        sel_idx.append(cluster_idx[np.argmin(d)])
    return [ts_list[i] for i in sel_idx], y.iloc[sel_idx]

### --- Custom binary serialization ---

def create_custom_binary(ts_list, output_path, compression_method='quantize', compression_params=None):
    compression_params = compression_params or {}
    compressed_data = []
    original_size = 0
    for ts in ts_list:
        original_size += len(ts) * 2
        if compression_method == "quantize":
            q, mn, mx = quantize(ts, compression_params.get('num_levels', 256))
            obj = {'data': q, 'min_val': mn, 'max_val': mx}
        elif compression_method == "wavelet":
            thr = compression_params.get('threshold_ratio', 0.1)
            wv = wavelet_compression(ts, threshold_ratio=thr)
            obj = {'data': wv}
        elif compression_method == "fourier":
            thr = compression_params.get('threshold_ratio', 0.1)
            ft = fourier_compression(ts, threshold_ratio=thr)
            obj = {'data': ft}
        else:
            obj = {'data': ts}
        compressed_data.append(obj)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('i', len(compressed_data)))
        f.write(struct.pack('i', len(compression_method)))
        f.write(compression_method.encode())
        for obj in compressed_data:
            s = pickle.dumps(obj)
            s = zlib.compress(s)
            f.write(struct.pack('i', len(s)))
            f.write(s)
    return os.path.getsize(output_path), original_size

def read_custom_binary(input_path):
    ts_list = []
    with open(input_path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        method_len = struct.unpack('i', f.read(4))[0]
        method = f.read(method_len).decode()
        for _ in range(n):
            size = struct.unpack('i', f.read(4))[0]
            s = f.read(size)
            obj = pickle.loads(zlib.decompress(s))
            if method == "quantize":
                ts = dequantize(obj['data'], obj['min_val'], obj['max_val'])
            else:
                ts = obj['data']
            ts_list.append(ts)
    return ts_list

### --- Reduction Workflow Example ---

def run_data_reduction_workflow():
    import matplotlib.pyplot as plt
    from data_loading import load_dataset

    # Load data
    X_train, y_train, _ = load_dataset('data')

    # Demonstrate and plot one sample with all compressions and save/print metrics
    sample = np.asarray(X_train[0])
    orig_size = sample.size * 2
    reduction_methods = [
        ("Downsample", downsample(sample,4)),
        ("PiecewiseConstant", piecewise_constant_approximation(sample, 50)),
        ("PiecewiseLinear", piecewise_linear_approximation(sample, 20)),
        ("Wavelet", wavelet_compression(sample, threshold_ratio=0.1)),
        ("Fourier", fourier_compression(sample, threshold_ratio=0.1)),
        ("Quantize", dequantize(*quantize(sample, 256))),
    ]
    metrics_list = []
    names = []
    print("=== Metrics for Single Sample (first train sample) ===")
    for method, arr in reduction_methods:
        met = compute_all_metrics(sample, arr, original_size=orig_size, compressed_size=arr.size*2)
        print(f"{method}: {met}")
        metrics_list.append(met)
        names.append(method)
    pd.DataFrame(metrics_list, index=names).to_csv(os.path.join(REDUCED_DIR, "sample_metrics.csv"))
    print(f"Saved sample_metrics.csv to {REDUCED_DIR}")

    plt.figure(figsize=(12,8))
    for i, (method, arr) in enumerate(reduction_methods):
        plt.subplot(2, 3, i+1)
        plt.plot(arr)
        plt.title(method)
    plt.tight_layout()
    plt.savefig(os.path.join(REDUCED_DIR, "reduction_examples.png"))
    print(f"Reduction plots saved to {REDUCED_DIR}")

    # Try 10%, 25%, 50% random and coreset, print and save compression results
    subset_metrics_rows = []
    for pct in [0.1, 0.25, 0.5]:
        for strat in ['random', 'kmeans']:
            print(f"\n==== {int(100*pct)}% {strat} subset ====")
            X_sub, y_sub = coreset_selection(X_train, y_train, ratio=pct, method=strat)
            print(f"Selected {len(X_sub)} samples.")
            # Save compressed version
            out_bin = os.path.join(REDUCED_DIR, f"train_{int(100*pct)}pct_{strat}.bin")
            size_c, size_o = create_custom_binary(X_sub, out_bin, compression_method='quantize')
            print(f"Saved compressed {out_bin} | orig MB: {size_o/1024/1024:.2f}, comp MB: {size_c/1024/1024:.2f}")
            # Compute metrics for first sample in reduced set
            recon_sub = read_custom_binary(out_bin)[0]
            orig_sub = np.array(X_sub[0])
            metrics = compute_all_metrics(orig_sub, recon_sub, original_size=len(orig_sub)*2, compressed_size=len(recon_sub)*2)
            print(f"First sample metrics ({int(100*pct)}% {strat}): {metrics}")
            # Save to results
            row = {
                "method": f"{int(100*pct)}pct_{strat}",
                "orig_mb": size_o/1024/1024,
                "comp_mb": size_c/1024/1024,
                **metrics
            }
            subset_metrics_rows.append(row)
    pd.DataFrame(subset_metrics_rows).to_csv(os.path.join(REDUCED_DIR, "subset_compression_metrics.csv"))
    print(f"Saved all subset compression metrics to {os.path.join(REDUCED_DIR, 'subset_compression_metrics.csv')}")

if __name__ == "__main__":
    run_data_reduction_workflow()
