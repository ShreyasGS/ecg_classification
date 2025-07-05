"""
metrics.py

Functions for evaluating data reduction techniques.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import data_loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_mae(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = np.array(original[:min_len])
    reconstructed = np.array(reconstructed[:min_len])
    return np.mean(np.abs(original - reconstructed))

def compute_mse(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = np.array(original[:min_len])
    reconstructed = np.array(reconstructed[:min_len])
    return np.mean((original - reconstructed) ** 2)

def compute_rmse(original, reconstructed):
    return np.sqrt(compute_mse(original, reconstructed))

def compute_normalized_mae(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = np.array(original[:min_len])
    reconstructed = np.array(reconstructed[:min_len])
    range_val = np.max(original) - np.min(original)
    if range_val > 0:
        return compute_mae(original, reconstructed) / range_val
    else:
        return 0

def compute_normalized_rmse(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = np.array(original[:min_len])
    reconstructed = np.array(reconstructed[:min_len])
    range_val = np.max(original) - np.min(original)
    if range_val > 0:
        return compute_rmse(original, reconstructed) / range_val
    else:
        return 0

def compute_prd(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = np.array(original[:min_len])
    reconstructed = np.array(reconstructed[:min_len])
    numerator = np.sum((original - reconstructed) ** 2)
    denominator = np.sum(original ** 2)
    if denominator > 0:
        return np.sqrt(numerator / denominator) * 100
    else:
        return 0

def compute_snr(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = np.array(original[:min_len])
    reconstructed = np.array(reconstructed[:min_len])
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return float('inf')

def compute_compression_ratio(original_size, compressed_size):
    if compressed_size > 0:
        return original_size / compressed_size
    else:
        return float('inf')

def compute_all_metrics(original, reconstructed, original_size=None, compressed_size=None):
    metrics = {
        'mae': compute_mae(original, reconstructed),
        'mse': compute_mse(original, reconstructed),
        'rmse': compute_rmse(original, reconstructed),
        'nmae': compute_normalized_mae(original, reconstructed),
        'nrmse': compute_normalized_rmse(original, reconstructed),
        'prd': compute_prd(original, reconstructed),
        'snr': compute_snr(original, reconstructed)
    }
    if original_size is not None and compressed_size is not None:
        metrics['compression_ratio'] = compute_compression_ratio(original_size, compressed_size)
    return metrics

def plot_metrics_comparison(method_names, metrics_list, metric_names=None, save_path=None):
    if metric_names is None:
        metric_names = ['mae', 'rmse', 'prd', 'snr', 'compression_ratio']
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 3 * len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]
    for i, metric_name in enumerate(metric_names):
        values = [metrics.get(metric_name, 0) for metrics in metrics_list]
        if metric_name == 'compression_ratio':
            axes[i].bar(method_names, values)
            axes[i].set_yscale('log')
        else:
            axes[i].bar(method_names, values)
        axes[i].set_title(f'{metric_name.upper()}')
        axes[i].set_xlabel('Method')
        axes[i].set_ylabel('Value')
        for j, value in enumerate(values):
            axes[i].text(j, value, f'{value:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_compression_vs_quality(method_names, metrics_list, quality_metric='rmse', save_path=None):
    compression_ratios = [metrics.get('compression_ratio', 1) for metrics in metrics_list]
    quality_values = [metrics.get(quality_metric, 0) for metrics in metrics_list]
    plt.figure(figsize=(10, 6))
    plt.scatter(compression_ratios, quality_values, s=100)
    for i, method_name in enumerate(method_names):
        plt.annotate(method_name, (compression_ratios[i], quality_values[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xscale('log')
    plt.xlabel('Compression Ratio (log scale)')
    plt.ylabel(f'{quality_metric.upper()}')
    plt.title(f'Compression Ratio vs. {quality_metric.upper()}')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_size_vs_quality(method_names, metrics_list, sizes, quality_metric='rmse', save_path=None):
    plt.figure(figsize=(10, 6))
    for i, method_name in enumerate(method_names):
        quality_values = [metrics_list[j][i].get(quality_metric, 0) for j in range(len(sizes))]
        plt.plot(sizes, quality_values, marker='o', label=method_name)
    plt.xlabel('Data Size (%)')
    plt.ylabel(f'{quality_metric.upper()}')
    plt.title(f'Data Size vs. {quality_metric.upper()}')
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_metrics_to_csv(method_names, metrics_list, output_path):
    df = pd.DataFrame(metrics_list)
    df['method'] = method_names
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

# Optionally, a main() function as in your script to demonstrate usage can be added here.
