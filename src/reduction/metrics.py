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

# Import data_loading only when needed to avoid circular imports
# from data_loading import read_binary_from


def compute_mae(original, reconstructed):
    """
    Compute Mean Absolute Error between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Mean Absolute Error.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    return np.mean(np.abs(original - reconstructed))


def compute_mse(original, reconstructed):
    """
    Compute Mean Squared Error between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Mean Squared Error.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    return np.mean(np.square(original - reconstructed))


def compute_rmse(original, reconstructed):
    """
    Compute Root Mean Squared Error between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Root Mean Squared Error.
    """
    return np.sqrt(compute_mse(original, reconstructed))


def compute_normalized_mae(original, reconstructed):
    """
    Compute Normalized Mean Absolute Error between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Normalized Mean Absolute Error.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Compute range
    range_val = np.max(original) - np.min(original)
    
    if range_val > 0:
        return compute_mae(original, reconstructed) / range_val
    else:
        return 0


def compute_normalized_rmse(original, reconstructed):
    """
    Compute Normalized Root Mean Squared Error between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Normalized Root Mean Squared Error.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Compute range
    range_val = np.max(original) - np.min(original)
    
    if range_val > 0:
        return compute_rmse(original, reconstructed) / range_val
    else:
        return 0


def compute_prd(original, reconstructed):
    """
    Compute Percentage Root-mean-square Difference between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Percentage Root-mean-square Difference.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Compute PRD
    numerator = np.sum(np.square(original - reconstructed))
    denominator = np.sum(np.square(original))
    
    if denominator > 0:
        return np.sqrt(numerator / denominator) * 100
    else:
        return 0


def compute_snr(original, reconstructed):
    """
    Compute Signal-to-Noise Ratio between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        
    Returns:
        float: Signal-to-Noise Ratio in dB.
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    # Compute SNR
    signal_power = np.sum(np.square(original))
    noise_power = np.sum(np.square(original - reconstructed))
    
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return float('inf')


def compute_compression_ratio(original_size, compressed_size):
    """
    Compute compression ratio.
    
    Args:
        original_size (int): Size of the original data in bytes.
        compressed_size (int): Size of the compressed data in bytes.
        
    Returns:
        float: Compression ratio.
    """
    if compressed_size > 0:
        return original_size / compressed_size
    else:
        return float('inf')


def compute_all_metrics(original, reconstructed, original_size=None, compressed_size=None):
    """
    Compute all metrics between original and reconstructed signals.
    
    Args:
        original (array): Original signal.
        reconstructed (array): Reconstructed signal.
        original_size (int, optional): Size of the original data in bytes.
        compressed_size (int, optional): Size of the compressed data in bytes.
        
    Returns:
        dict: Dictionary of metrics.
    """
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
    """
    Plot comparison of metrics for different methods.
    
    Args:
        method_names (list): List of method names.
        metrics_list (list): List of metrics dictionaries.
        metric_names (list, optional): List of metric names to plot.
        save_path (str, optional): Path to save the plot.
    """
    if metric_names is None:
        metric_names = ['mae', 'rmse', 'prd', 'snr', 'compression_ratio']
    
    # Create figure
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 3 * len(metric_names)))
    
    # Ensure axes is a list
    if len(metric_names) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        values = [metrics.get(metric_name, 0) for metrics in metrics_list]
        
        # Special case for compression ratio
        if metric_name == 'compression_ratio':
            # Use log scale for compression ratio
            axes[i].bar(method_names, values)
            axes[i].set_yscale('log')
        else:
            axes[i].bar(method_names, values)
        
        axes[i].set_title(f'{metric_name.upper()}')
        axes[i].set_xlabel('Method')
        axes[i].set_ylabel('Value')
        
        # Add values on top of bars
        for j, value in enumerate(values):
            axes[i].text(j, value, f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_compression_vs_quality(method_names, metrics_list, quality_metric='rmse', save_path=None):
    """
    Plot compression ratio vs. quality metric.
    
    Args:
        method_names (list): List of method names.
        metrics_list (list): List of metrics dictionaries.
        quality_metric (str): Quality metric to use.
        save_path (str, optional): Path to save the plot.
    """
    # Extract metrics
    compression_ratios = [metrics.get('compression_ratio', 1) for metrics in metrics_list]
    quality_values = [metrics.get(quality_metric, 0) for metrics in metrics_list]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot scatter points
    plt.scatter(compression_ratios, quality_values, s=100)
    
    # Add method names as labels
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
    """
    Plot data size vs. quality metric for different methods.
    
    Args:
        method_names (list): List of method names.
        metrics_list (list): List of lists of metrics dictionaries for different sizes.
        sizes (list): List of data sizes.
        quality_metric (str): Quality metric to use.
        save_path (str, optional): Path to save the plot.
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot line for each method
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
    """
    Save metrics to a CSV file.
    
    Args:
        method_names (list): List of method names.
        metrics_list (list): List of metrics dictionaries.
        output_path (str): Path to save the CSV file.
    """
    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Add method names
    df['method'] = method_names
    
    # Reorder columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def main():
    """
    Main function to demonstrate metrics computation.
    """
    import sys
    import os
    
    # Add parent directory to path to import data_loading
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from data_loading import load_dataset
        from reduction.reduce import (
            downsample, piecewise_constant_approximation, piecewise_linear_approximation,
            wavelet_compression, fourier_compression
        )
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all required packages are installed.")
        return
    
    # Load data
    try:
        X_train, y_train, _ = load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic data for demonstration...")
        # Create synthetic data
        import numpy as np
        X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
        y_train = None
    
    if X_train is None:
        print("Could not load data. Creating synthetic data...")
        # Create synthetic data
        import numpy as np
        X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
    
    # Select a sample
    sample_idx = 0
    sample = X_train[sample_idx]
    
    try:
        # Apply different reduction techniques
        methods = {
            'Downsampling': downsample(sample, factor=4),
            'Piecewise Constant': piecewise_constant_approximation(sample, num_segments=50),
            'Piecewise Linear': piecewise_linear_approximation(sample, num_segments=20),
            'Wavelet': wavelet_compression(sample, threshold_ratio=0.1),
            'Fourier': fourier_compression(sample, threshold_ratio=0.1)
        }
        
        # Compute metrics
        method_names = []
        metrics_list = []
        
        for method_name, reconstructed in methods.items():
            method_names.append(method_name)
            
            # Compute original and compressed sizes
            original_size = len(sample) * 2  # 2 bytes per value (16-bit)
            compressed_size = len(reconstructed) * 2  # Simplified assumption
            
            # Compute metrics
            metrics = compute_all_metrics(sample, reconstructed, original_size, compressed_size)
            metrics_list.append(metrics)
            
            # Print metrics
            print(f"\n{method_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # Plot metrics comparison
        try:
            plot_metrics_comparison(method_names, metrics_list, save_path="metrics_comparison.png")
            print("Metrics comparison plot saved to 'metrics_comparison.png'")
            
            # Plot compression vs. quality
            plot_compression_vs_quality(method_names, metrics_list, save_path="compression_vs_quality.png")
            print("Compression vs. quality plot saved to 'compression_vs_quality.png'")
            
            # Save metrics to CSV
            save_metrics_to_csv(method_names, metrics_list, "reduction_metrics.csv")
        except Exception as e:
            print(f"Error creating plots or saving metrics: {e}")
    
    except Exception as e:
        print(f"Error applying reduction techniques: {e}")
        print("Please ensure all required packages are installed.")


if __name__ == "__main__":
    main() 