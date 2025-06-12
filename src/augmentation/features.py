"""
features.py

Functions for extracting features from ECG time series data.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
import pywt
# Comment out problematic imports
# import biosppy.signals.ecg as ecg
# import neurokit2 as nk
import sys
import os

# Add parent directory to path to import data_loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_statistical_features(time_series):
    """
    Extract statistical features from a time series.
    
    Args:
        time_series (array): Time series data.
        
    Returns:
        dict: Dictionary of features.
    """
    features = {
        'mean': np.mean(time_series),
        'std': np.std(time_series),
        'min': np.min(time_series),
        'max': np.max(time_series),
        'median': np.median(time_series),
        'q1': np.percentile(time_series, 25),
        'q3': np.percentile(time_series, 75),
        'iqr': np.percentile(time_series, 75) - np.percentile(time_series, 25),
        'skewness': stats.skew(time_series),
        'kurtosis': stats.kurtosis(time_series),
        'rms': np.sqrt(np.mean(np.square(time_series))),
        'energy': np.sum(np.square(time_series)),
        'abs_energy': np.sum(np.abs(time_series)),
        'mean_abs_change': np.mean(np.abs(np.diff(time_series))),
        'mean_change': np.mean(np.diff(time_series)),
        'abs_max': np.max(np.abs(time_series)),
        'count_above_mean': np.sum(time_series > np.mean(time_series)),
        'count_below_mean': np.sum(time_series < np.mean(time_series)),
        'range': np.max(time_series) - np.min(time_series),
        'var': np.var(time_series)
    }
    
    return features


def extract_frequency_features(time_series, fs=300):
    """
    Extract frequency domain features from a time series.
    
    Args:
        time_series (array): Time series data.
        fs (int): Sampling frequency.
        
    Returns:
        dict: Dictionary of features.
    """
    # Compute FFT
    fft = np.fft.rfft(time_series)
    fft_mag = np.abs(fft)
    fft_phase = np.angle(fft)
    
    # Compute frequency axis
    freqs = np.fft.rfftfreq(len(time_series), d=1/fs)
    
    # Compute power spectral density
    psd = fft_mag ** 2 / len(time_series)
    
    # Extract features
    features = {
        'fft_mean': np.mean(fft_mag),
        'fft_std': np.std(fft_mag),
        'fft_max': np.max(fft_mag),
        'fft_min': np.min(fft_mag),
        'fft_median': np.median(fft_mag),
        'fft_skewness': stats.skew(fft_mag),
        'fft_kurtosis': stats.kurtosis(fft_mag),
        'fft_energy': np.sum(fft_mag ** 2),
        'fft_entropy': stats.entropy(fft_mag + 1e-10),
        'spectral_centroid': np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0,
        'spectral_spread': np.sqrt(np.sum(((freqs - np.sum(freqs * psd) / np.sum(psd)) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0,
        'spectral_skewness': stats.skew(psd),
        'spectral_kurtosis': stats.kurtosis(psd),
        'spectral_slope': stats.linregress(freqs, psd)[0] if len(freqs) > 1 else 0,
        'spectral_rolloff': freqs[np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0][0]] if len(np.where(np.cumsum(psd) >= 0.95 * np.sum(psd))[0]) > 0 else 0,
        'spectral_flatness': stats.gmean(psd + 1e-10) / (np.mean(psd) + 1e-10)
    }
    
    # Compute power in different frequency bands
    bands = [(0, 5), (5, 15), (15, 30), (30, 50), (50, 100), (100, 150)]
    
    for i, (low, high) in enumerate(bands):
        band_indices = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.sum(psd[band_indices])
        features[f'band_power_{low}_{high}'] = band_power
    
    return features


def extract_wavelet_features(time_series, wavelet='db4', level=5):
    """
    Extract wavelet features from a time series.
    
    Args:
        time_series (array): Time series data.
        wavelet (str): Wavelet type.
        level (int): Decomposition level.
        
    Returns:
        dict: Dictionary of features.
    """
    # Compute wavelet coefficients
    coeffs = pywt.wavedec(time_series, wavelet, level=level)
    
    features = {}
    
    # Extract features from each level
    for i, coeff in enumerate(coeffs):
        if i == 0:
            name = 'approximation'
        else:
            name = f'detail_{i}'
        
        features[f'{name}_mean'] = np.mean(coeff)
        features[f'{name}_std'] = np.std(coeff)
        features[f'{name}_max'] = np.max(coeff)
        features[f'{name}_min'] = np.min(coeff)
        features[f'{name}_energy'] = np.sum(coeff ** 2)
        features[f'{name}_entropy'] = stats.entropy(np.abs(coeff) + 1e-10)
    
    return features


def extract_ecg_features(time_series, fs=300):
    """
    Extract ECG-specific features from a time series.
    
    Args:
        time_series (array): Time series data.
        fs (int): Sampling frequency.
        
    Returns:
        dict: Dictionary of features.
    """
    try:
        # Try to import biosppy
        try:
            import biosppy.signals.ecg as ecg
        except ImportError:
            print("Warning: biosppy not installed. Using fallback implementation for ECG features.")
            # Fallback implementation without biosppy
            return extract_ecg_features_fallback(time_series, fs)
            
        # Process ECG signal using BioSPPy
        out = ecg.ecg(signal=time_series, sampling_rate=fs, show=False)
        
        # Extract R-peaks
        rpeaks = out['rpeaks']
        
        if len(rpeaks) < 2:
            # Not enough R-peaks detected
            return {
                'heart_rate': 0,
                'rr_mean': 0,
                'rr_std': 0,
                'rr_median': 0,
                'rr_range': 0,
                'rr_pnn50': 0,
                'rr_nn50': 0,
                'rr_rmssd': 0,
                'rr_sdsd': 0,
                'r_peak_count': 0
            }
        
        # Calculate RR intervals
        rr_intervals = np.diff(rpeaks) / fs * 1000  # in ms
        
        # Calculate heart rate
        heart_rate = 60 / (np.mean(rr_intervals) / 1000)
        
        # Calculate heart rate variability metrics
        nn50 = sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = nn50 / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        sdsd = np.std(np.diff(rr_intervals))
        
        features = {
            'heart_rate': heart_rate,
            'rr_mean': np.mean(rr_intervals),
            'rr_std': np.std(rr_intervals),
            'rr_median': np.median(rr_intervals),
            'rr_range': np.max(rr_intervals) - np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
            'rr_pnn50': pnn50,
            'rr_nn50': nn50,
            'rr_rmssd': rmssd,
            'rr_sdsd': sdsd,
            'r_peak_count': len(rpeaks)
        }
        
        return features
    except Exception as e:
        print(f"Error extracting ECG features: {e}")
        return extract_ecg_features_fallback(time_series, fs)


def extract_ecg_features_fallback(time_series, fs=300):
    """
    Fallback implementation for ECG feature extraction when biosppy is not available.
    
    Args:
        time_series (array): Time series data.
        fs (int): Sampling frequency.
        
    Returns:
        dict: Dictionary of features.
    """
    # Simple peak detection using threshold
    threshold = np.mean(time_series) + 1.5 * np.std(time_series)
    above_threshold = time_series > threshold
    
    # Find transitions from below to above threshold
    transitions = np.diff(above_threshold.astype(int))
    rpeaks = np.where(transitions == 1)[0]
    
    if len(rpeaks) < 2:
        # Not enough R-peaks detected
        return {
            'heart_rate': 0,
            'rr_mean': 0,
            'rr_std': 0,
            'rr_median': 0,
            'rr_range': 0,
            'rr_pnn50': 0,
            'rr_nn50': 0,
            'rr_rmssd': 0,
            'rr_sdsd': 0,
            'r_peak_count': 0
        }
    
    # Calculate RR intervals
    rr_intervals = np.diff(rpeaks) / fs * 1000  # in ms
    
    # Calculate heart rate
    heart_rate = 60 / (np.mean(rr_intervals) / 1000)
    
    # Calculate heart rate variability metrics
    nn50 = sum(np.abs(np.diff(rr_intervals)) > 50)
    pnn50 = nn50 / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    sdsd = np.std(np.diff(rr_intervals))
    
    features = {
        'heart_rate': heart_rate,
        'rr_mean': np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'rr_median': np.median(rr_intervals),
        'rr_range': np.max(rr_intervals) - np.min(rr_intervals) if len(rr_intervals) > 0 else 0,
        'rr_pnn50': pnn50,
        'rr_nn50': nn50,
        'rr_rmssd': rmssd,
        'rr_sdsd': sdsd,
        'r_peak_count': len(rpeaks)
    }
    
    return features


def extract_neurokit_features(time_series, fs=300):
    """
    Extract ECG features using NeuroKit2.
    
    Args:
        time_series (array): Time series data.
        fs (int): Sampling frequency.
        
    Returns:
        dict: Dictionary of features.
    """
    try:
        # Try to import neurokit2
        try:
            import neurokit2 as nk
        except ImportError:
            print("Warning: neurokit2 not installed. Skipping NeuroKit features.")
            return {}
            
        # Process ECG signal using NeuroKit2
        signals, info = nk.ecg_process(time_series, sampling_rate=fs)
        
        # Extract heart rate variability
        hrv = nk.hrv(info['ECG_R_Peaks'], sampling_rate=fs, show=False)
        
        # Extract features
        features = {}
        
        # Add HRV features
        for col in hrv.columns:
            if col != 'HRV_ULF' and col != 'HRV_VLF':  # Skip some features that might be NaN
                features[col] = hrv[col].values[0]
        
        # Add quality metrics
        features['ecg_quality'] = np.mean(signals['ECG_Quality'])
        
        return features
    
    except Exception as e:
        # Return empty dict if NeuroKit processing fails
        print(f"Error extracting NeuroKit features: {e}")
        return {}


def extract_all_features(time_series, fs=300):
    """
    Extract all features from a time series.
    
    Args:
        time_series (array): Time series data.
        fs (int): Sampling frequency.
        
    Returns:
        dict: Dictionary of features.
    """
    # Extract all feature types
    statistical_features = extract_statistical_features(time_series)
    frequency_features = extract_frequency_features(time_series, fs)
    wavelet_features = extract_wavelet_features(time_series)
    ecg_features = extract_ecg_features(time_series, fs)
    
    # Combine all features
    all_features = {}
    all_features.update(statistical_features)
    all_features.update(frequency_features)
    all_features.update(wavelet_features)
    all_features.update(ecg_features)
    
    # Try to extract NeuroKit features
    neurokit_features = extract_neurokit_features(time_series, fs)
    all_features.update(neurokit_features)
    
    return all_features


def extract_features_from_dataset(X, fs=300, feature_types=None):
    """
    Extract features from a dataset of time series.
    
    Args:
        X (list): List of time series.
        fs (int): Sampling frequency.
        feature_types (list): List of feature types to extract.
        
    Returns:
        pandas.DataFrame: DataFrame of features.
    """
    feature_extractors = {
        'statistical': extract_statistical_features,
        'frequency': lambda ts: extract_frequency_features(ts, fs),
        'wavelet': extract_wavelet_features,
        'ecg': lambda ts: extract_ecg_features(ts, fs),
        'neurokit': lambda ts: extract_neurokit_features(ts, fs),
        'all': lambda ts: extract_all_features(ts, fs)
    }
    
    if feature_types is None:
        feature_types = ['statistical', 'frequency', 'wavelet', 'ecg']
    
    all_features = []
    
    for i, ts in enumerate(X):
        if i % 100 == 0:
            print(f"Extracting features for sample {i}/{len(X)}...")
        
        # Extract features
        sample_features = {}
        
        for feature_type in feature_types:
            if feature_type in feature_extractors:
                features = feature_extractors[feature_type](ts)
                sample_features.update(features)
        
        all_features.append(sample_features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    return df


def main():
    """
    Main function to demonstrate feature extraction.
    """
    import matplotlib.pyplot as plt
    import sys
    import os
    
    # Add parent directory to path to import data_loading
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from data_loading import load_dataset
    except ImportError as e:
        print(f"Error importing data_loading: {e}")
        print("Creating synthetic data for demonstration...")
        # Create synthetic data
        X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
        y_train = pd.DataFrame({'label': [0]})
    else:
        # Load data
        try:
            X_train, y_train, _ = load_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating synthetic data for demonstration...")
            # Create synthetic data
            X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
            y_train = pd.DataFrame({'label': [0]})
    
    if X_train is None:
        print("Could not load data. Creating synthetic data...")
        # Create synthetic data
        X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
        y_train = pd.DataFrame({'label': [0]})
    
    # Select a sample
    sample_idx = 0
    sample = X_train[sample_idx]
    
    try:
        # Extract features
        print("Extracting statistical features...")
        statistical_features = extract_statistical_features(sample)
        
        print("Extracting frequency features...")
        frequency_features = extract_frequency_features(sample)
        
        print("Extracting wavelet features...")
        wavelet_features = extract_wavelet_features(sample)
        
        print("Extracting ECG features...")
        ecg_features = extract_ecg_features(sample)
        
        # Print features
        print("\nStatistical features:")
        for name, value in statistical_features.items():
            print(f"  {name}: {value:.4f}")
        
        print("\nFrequency features:")
        for name, value in frequency_features.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
        
        print("\nWavelet features (first 5):")
        for i, (name, value) in enumerate(wavelet_features.items()):
            if i < 5 and isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
        
        print("\nECG features:")
        for name, value in ecg_features.items():
            print(f"  {name}: {value:.4f}")
        
        # Try to extract NeuroKit features if available
        try:
            print("\nExtracting NeuroKit features...")
            neurokit_features = extract_neurokit_features(sample)
            
            if neurokit_features:
                print("\nNeuroKit features (first 5):")
                for i, (name, value) in enumerate(neurokit_features.items()):
                    if i < 5 and isinstance(value, (int, float)):
                        print(f"  {name}: {value:.4f}")
        except Exception as e:
            print(f"Error extracting NeuroKit features: {e}")
        
        # Plot sample and features
        plt.figure(figsize=(15, 10))
        
        # Plot original signal
        plt.subplot(3, 1, 1)
        plt.plot(sample)
        plt.title("Original Signal")
        
        # Plot frequency spectrum
        plt.subplot(3, 1, 2)
        fft = np.abs(np.fft.rfft(sample))
        freqs = np.fft.rfftfreq(len(sample), d=1/300)
        plt.plot(freqs, fft)
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        
        # Plot wavelet coefficients
        plt.subplot(3, 1, 3)
        coeffs = pywt.wavedec(sample, 'db4', level=5)
        plt.plot(pywt.waverec(coeffs, 'db4'))
        plt.title("Wavelet Reconstruction")
        
        plt.tight_layout()
        plt.savefig("feature_extraction_example.png")
        plt.close()
        
        print("\nFeature extraction example saved to 'feature_extraction_example.png'")
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")


if __name__ == "__main__":
    main() 