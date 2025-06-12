"""
reduce.py

Functions for reducing ECG time series data size.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pywt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import pickle
import zlib
import struct
import os
import sys

# Add parent directory to path to import data_loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import data_loading only when needed to avoid circular imports
# from data_loading import read_binary_from


def downsample(time_series, factor=2):
    """
    Downsample a time series by a given factor.
    
    Args:
        time_series (array): Time series to downsample.
        factor (int): Downsampling factor.
        
    Returns:
        array: Downsampled time series.
    """
    return time_series[::factor]


def piecewise_constant_approximation(time_series, num_segments):
    """
    Approximate a time series using piecewise constant approximation.
    
    Args:
        time_series (array): Time series to approximate.
        num_segments (int): Number of segments.
        
    Returns:
        array: Approximated time series.
    """
    # Calculate segment length
    segment_length = len(time_series) // num_segments
    
    # Initialize approximated time series
    approximated = np.zeros(len(time_series))
    
    # Compute mean for each segment
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else len(time_series)
        segment_mean = np.mean(time_series[start:end])
        approximated[start:end] = segment_mean
    
    return approximated


def piecewise_linear_approximation(time_series, num_segments):
    """
    Approximate a time series using piecewise linear approximation.
    
    Args:
        time_series (array): Time series to approximate.
        num_segments (int): Number of segments.
        
    Returns:
        array: Approximated time series.
    """
    # Calculate segment length
    segment_length = len(time_series) // num_segments
    
    # Initialize approximated time series
    approximated = np.zeros(len(time_series))
    
    # Compute linear approximation for each segment
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else len(time_series)
        
        if end - start <= 1:
            # If segment has only one point, use constant approximation
            approximated[start:end] = time_series[start]
        else:
            # Compute linear approximation
            x = np.arange(start, end)
            y = time_series[start:end]
            coeffs = np.polyfit(x, y, 1)
            approximated[start:end] = np.polyval(coeffs, x)
    
    return approximated


def wavelet_compression(time_series, wavelet='db4', threshold_ratio=0.1):
    """
    Compress a time series using wavelet transform.
    
    Args:
        time_series (array): Time series to compress.
        wavelet (str): Wavelet type.
        threshold_ratio (float): Ratio of coefficients to keep.
        
    Returns:
        array: Compressed time series.
    """
    # Compute wavelet transform
    coeffs = pywt.wavedec(time_series, wavelet)
    
    # Flatten coefficients
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Sort coefficients by magnitude
    sorted_indices = np.argsort(np.abs(coeff_arr))
    
    # Set small coefficients to zero
    threshold_idx = int(len(sorted_indices) * (1 - threshold_ratio))
    coeff_arr[sorted_indices[:threshold_idx]] = 0
    
    # Convert back to coefficients
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec')
    
    # Reconstruct signal
    reconstructed = pywt.waverec(coeffs, wavelet)
    
    # Ensure the reconstructed signal has the same length as the original
    if len(reconstructed) > len(time_series):
        reconstructed = reconstructed[:len(time_series)]
    
    return reconstructed


def fourier_compression(time_series, threshold_ratio=0.1):
    """
    Compress a time series using Fourier transform.
    
    Args:
        time_series (array): Time series to compress.
        threshold_ratio (float): Ratio of coefficients to keep.
        
    Returns:
        array: Compressed time series.
    """
    # Compute FFT
    fft = np.fft.rfft(time_series)
    
    # Sort coefficients by magnitude
    sorted_indices = np.argsort(np.abs(fft))
    
    # Set small coefficients to zero
    threshold_idx = int(len(sorted_indices) * (1 - threshold_ratio))
    fft[sorted_indices[:threshold_idx]] = 0
    
    # Reconstruct signal
    reconstructed = np.fft.irfft(fft, n=len(time_series))
    
    return reconstructed


def quantize(time_series, num_levels=256):
    """
    Quantize a time series to a given number of levels.
    
    Args:
        time_series (array): Time series to quantize.
        num_levels (int): Number of quantization levels.
        
    Returns:
        tuple: (quantized time series, min value, max value)
    """
    # Get min and max values
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    
    # Quantize
    quantized = np.round((time_series - min_val) / (max_val - min_val) * (num_levels - 1))
    
    return quantized, min_val, max_val


def dequantize(quantized, min_val, max_val, num_levels=256):
    """
    Dequantize a time series.
    
    Args:
        quantized (array): Quantized time series.
        min_val (float): Minimum value of the original time series.
        max_val (float): Maximum value of the original time series.
        num_levels (int): Number of quantization levels.
        
    Returns:
        array: Dequantized time series.
    """
    return quantized / (num_levels - 1) * (max_val - min_val) + min_val


def delta_encoding(time_series):
    """
    Encode a time series using delta encoding.
    
    Args:
        time_series (array): Time series to encode.
        
    Returns:
        tuple: (encoded time series, first value)
    """
    # Get first value
    first_val = time_series[0]
    
    # Compute deltas
    deltas = np.diff(time_series)
    
    return deltas, first_val


def delta_decoding(deltas, first_val):
    """
    Decode a delta-encoded time series.
    
    Args:
        deltas (array): Delta-encoded time series.
        first_val (float): First value of the original time series.
        
    Returns:
        array: Decoded time series.
    """
    # Initialize decoded time series
    decoded = np.zeros(len(deltas) + 1)
    decoded[0] = first_val
    
    # Reconstruct signal
    for i in range(len(deltas)):
        decoded[i + 1] = decoded[i] + deltas[i]
    
    return decoded


def run_length_encoding(time_series):
    """
    Encode a time series using run-length encoding.
    
    Args:
        time_series (array): Time series to encode.
        
    Returns:
        list: List of (value, count) tuples.
    """
    # Initialize encoded time series
    encoded = []
    
    # Current value and count
    current_val = time_series[0]
    count = 1
    
    # Encode
    for i in range(1, len(time_series)):
        if time_series[i] == current_val:
            count += 1
        else:
            encoded.append((current_val, count))
            current_val = time_series[i]
            count = 1
    
    # Add last run
    encoded.append((current_val, count))
    
    return encoded


def run_length_decoding(encoded):
    """
    Decode a run-length encoded time series.
    
    Args:
        encoded (list): List of (value, count) tuples.
        
    Returns:
        array: Decoded time series.
    """
    # Initialize decoded time series
    decoded = []
    
    # Decode
    for val, count in encoded:
        decoded.extend([val] * count)
    
    return np.array(decoded)


def coreset_selection(time_series_list, ratio=0.1, method='kmeans'):
    """
    Select a coreset from a list of time series.
    
    Args:
        time_series_list (list): List of time series.
        ratio (float): Ratio of time series to select.
        method (str): Method to use for selection ('kmeans' or 'random').
        
    Returns:
        list: Selected time series.
    """
    # Number of time series to select
    num_select = max(1, int(len(time_series_list) * ratio))
    
    if method == 'random':
        # Random selection
        indices = random.sample(range(len(time_series_list)), num_select)
        selected = [time_series_list[i] for i in indices]
    elif method == 'kmeans':
        # Extract features for clustering
        features = []
        for ts in time_series_list:
            # Use simple statistics as features
            features.append([
                np.mean(ts),
                np.std(ts),
                np.min(ts),
                np.max(ts),
                np.median(ts)
            ])
        
        # Normalize features
        features = np.array(features)
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Cluster time series
        kmeans = KMeans(n_clusters=num_select, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Select time series closest to cluster centers
        selected = []
        for i in range(num_select):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                # Find time series closest to cluster center
                cluster_features = features[cluster_indices]
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected.append(time_series_list[closest_idx])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return selected


def create_custom_binary(time_series_list, output_path, compression_method='quantize', compression_params=None):
    """
    Create a custom binary file from a list of time series.
    
    Args:
        time_series_list (list): List of time series.
        output_path (str): Path to save the binary file.
        compression_method (str): Compression method to use.
        compression_params (dict): Parameters for the compression method.
        
    Returns:
        tuple: (compressed_size, original_size)
    """
    compression_params = compression_params or {}
    
    # Compress each time series
    compressed_data = []
    original_size = 0
    
    for ts in time_series_list:
        original_size += len(ts) * 2  # 2 bytes per value (16-bit)
        
        if compression_method == 'quantize':
            num_levels = compression_params.get('num_levels', 256)
            quantized, min_val, max_val = quantize(ts, num_levels)
            
            # Store as 8-bit integers
            compressed = {
                'data': quantized.astype(np.uint8),
                'min_val': min_val,
                'max_val': max_val
            }
        elif compression_method == 'wavelet':
            threshold_ratio = compression_params.get('threshold_ratio', 0.1)
            wavelet = compression_params.get('wavelet', 'db4')
            compressed_ts = wavelet_compression(ts, wavelet, threshold_ratio)
            
            # Store as 16-bit integers
            compressed = {
                'data': compressed_ts.astype(np.int16)
            }
        elif compression_method == 'fourier':
            threshold_ratio = compression_params.get('threshold_ratio', 0.1)
            compressed_ts = fourier_compression(ts, threshold_ratio)
            
            # Store as 16-bit integers
            compressed = {
                'data': compressed_ts.astype(np.int16)
            }
        elif compression_method == 'delta':
            deltas, first_val = delta_encoding(ts)
            
            # Store as 8-bit integers with scaling
            max_delta = np.max(np.abs(deltas))
            scale = 127 / max_delta if max_delta > 0 else 1
            scaled_deltas = np.round(deltas * scale).astype(np.int8)
            
            compressed = {
                'data': scaled_deltas,
                'first_val': first_val,
                'scale': scale
            }
        elif compression_method == 'pca':
            # Use PCA for dimensionality reduction
            n_components = compression_params.get('n_components', 10)
            pca = PCA(n_components=n_components)
            
            # Reshape for PCA
            ts_reshaped = ts.reshape(1, -1)
            
            # Apply PCA
            transformed = pca.fit_transform(ts_reshaped)
            
            compressed = {
                'transformed': transformed,
                'components': pca.components_,
                'mean': pca.mean_
            }
        else:
            # No compression
            compressed = {
                'data': ts.astype(np.int16)
            }
        
        compressed_data.append(compressed)
    
    # Save to binary file
    with open(output_path, 'wb') as f:
        # Write number of time series
        f.write(struct.pack('i', len(compressed_data)))
        
        # Write compression method
        f.write(struct.pack('i', len(compression_method)))
        f.write(compression_method.encode())
        
        # Write each compressed time series
        for compressed in compressed_data:
            # Serialize the compressed data
            serialized = pickle.dumps(compressed)
            
            # Compress with zlib
            compressed_bytes = zlib.compress(serialized)
            
            # Write size and data
            f.write(struct.pack('i', len(compressed_bytes)))
            f.write(compressed_bytes)
    
    # Get compressed size
    compressed_size = os.path.getsize(output_path)
    
    return compressed_size, original_size


def read_custom_binary(input_path):
    """
    Read a custom binary file.
    
    Args:
        input_path (str): Path to the binary file.
        
    Returns:
        list: List of time series.
    """
    time_series_list = []
    
    with open(input_path, 'rb') as f:
        # Read number of time series
        num_time_series = struct.unpack('i', f.read(4))[0]
        
        # Read compression method
        method_len = struct.unpack('i', f.read(4))[0]
        compression_method = f.read(method_len).decode()
        
        # Read each compressed time series
        for _ in range(num_time_series):
            # Read size and data
            size = struct.unpack('i', f.read(4))[0]
            compressed_bytes = f.read(size)
            
            # Decompress with zlib
            serialized = zlib.decompress(compressed_bytes)
            
            # Deserialize the compressed data
            compressed = pickle.loads(serialized)
            
            # Decompress based on method
            if compression_method == 'quantize':
                ts = dequantize(compressed['data'], compressed['min_val'], compressed['max_val'])
            elif compression_method == 'wavelet' or compression_method == 'fourier':
                ts = compressed['data']
            elif compression_method == 'delta':
                deltas = compressed['data'] / compressed['scale']
                ts = delta_decoding(deltas, compressed['first_val'])
            elif compression_method == 'pca':
                # Reconstruct from PCA
                ts = np.dot(compressed['transformed'], compressed['components']) + compressed['mean']
                ts = ts.flatten()
            else:
                # No compression
                ts = compressed['data']
            
            time_series_list.append(ts)
    
    return time_series_list


def evaluate_compression(original, compressed):
    """
    Evaluate compression quality.
    
    Args:
        original (array): Original time series.
        compressed (array): Compressed time series.
        
    Returns:
        dict: Dictionary of metrics.
    """
    # Ensure same length
    min_len = min(len(original), len(compressed))
    original = original[:min_len]
    compressed = compressed[:min_len]
    
    # Compute metrics
    mae = np.mean(np.abs(original - compressed))
    mse = np.mean(np.square(original - compressed))
    rmse = np.sqrt(mse)
    
    # Normalized metrics
    range_val = np.max(original) - np.min(original)
    if range_val > 0:
        nmae = mae / range_val
        nrmse = rmse / range_val
    else:
        nmae = 0
        nrmse = 0
    
    # Compression ratio
    original_size = len(original) * 2  # 2 bytes per value (16-bit)
    compressed_size = len(compressed) * 2  # Depends on the compression method
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'nmae': nmae,
        'nrmse': nrmse,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size > 0 else float('inf')
    }
    
    return metrics


def main():
    """
    Main function to demonstrate data reduction.
    """
    import matplotlib.pyplot as plt
    from data_loading import load_dataset
    
    # Load data
    X_train, y_train, _ = load_dataset()
    
    if X_train is None:
        print("Could not load data. Exiting.")
        return
    
    # Select a sample
    sample_idx = 0
    sample = X_train[sample_idx]
    label = y_train.iloc[sample_idx]['label']
    
    print(f"Original sample (class {label}):")
    print(f"Length: {len(sample)}")
    
    # Apply different reduction techniques
    plt.figure(figsize=(15, 10))
    
    # Plot original
    plt.subplot(3, 2, 1)
    plt.plot(sample)
    plt.title("Original")
    
    # Downsample
    factor = 4
    downsampled = downsample(sample, factor)
    plt.subplot(3, 2, 2)
    plt.plot(np.arange(0, len(sample), factor), downsampled)
    plt.title(f"Downsampled (factor={factor})")
    
    # Piecewise constant approximation
    num_segments = 50
    pca = piecewise_constant_approximation(sample, num_segments)
    plt.subplot(3, 2, 3)
    plt.plot(pca)
    plt.title(f"Piecewise Constant (segments={num_segments})")
    
    # Piecewise linear approximation
    num_segments = 20
    pla = piecewise_linear_approximation(sample, num_segments)
    plt.subplot(3, 2, 4)
    plt.plot(pla)
    plt.title(f"Piecewise Linear (segments={num_segments})")
    
    # Wavelet compression
    threshold_ratio = 0.1
    wavelet_compressed = wavelet_compression(sample, threshold_ratio=threshold_ratio)
    plt.subplot(3, 2, 5)
    plt.plot(wavelet_compressed)
    plt.title(f"Wavelet Compression (ratio={threshold_ratio})")
    
    # Fourier compression
    threshold_ratio = 0.1
    fourier_compressed = fourier_compression(sample, threshold_ratio=threshold_ratio)
    plt.subplot(3, 2, 6)
    plt.plot(fourier_compressed)
    plt.title(f"Fourier Compression (ratio={threshold_ratio})")
    
    plt.tight_layout()
    plt.savefig("reduction_examples.png")
    plt.close()
    
    print("Reduction examples saved to 'reduction_examples.png'")
    
    # Evaluate compression quality
    print("\nCompression quality:")
    
    techniques = [
        ("Downsampling", downsampled),
        ("Piecewise Constant", pca),
        ("Piecewise Linear", pla),
        ("Wavelet", wavelet_compressed),
        ("Fourier", fourier_compressed)
    ]
    
    for name, compressed in techniques:
        metrics = evaluate_compression(sample, compressed)
        print(f"\n{name}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Compression ratio: {metrics['compression_ratio']:.2f}")
    
    # Create custom binary file
    print("\nCreating custom binary file...")
    
    # Use a subset of the data
    subset_size = min(100, len(X_train))
    subset = X_train[:subset_size]
    
    # Create binary file with different compression methods
    methods = ['quantize', 'wavelet', 'fourier', 'delta', 'pca']
    
    for method in methods:
        output_path = f"X_train_reduced_{method}.bin"
        compressed_size, original_size = create_custom_binary(subset, output_path, method)
        
        print(f"\n{method.capitalize()} compression:")
        print(f"  Original size: {original_size} bytes")
        print(f"  Compressed size: {compressed_size} bytes")
        print(f"  Compression ratio: {original_size / compressed_size:.2f}")
        
        # Read back
        reconstructed = read_custom_binary(output_path)
        
        # Evaluate quality
        metrics = evaluate_compression(subset[0], reconstructed[0])
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main() 