"""
augment.py

Functions for augmenting ECG time series data.
"""
import numpy as np
# Conditionally import torch to handle case when it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some augmentation functions will be limited.")
from scipy import signal
from scipy.interpolate import interp1d
import random
import sys
import os

# Add parent directory to path to import data_loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TimeSeriesAugmenter:
    """
    Class for applying various augmentation techniques to time series data.
    """
    def __init__(self, augmentations=None, p=0.5):
        """
        Initialize the augmenter.
        
        Args:
            augmentations (list): List of augmentation functions to apply.
            p (float): Probability of applying each augmentation.
        """
        self.augmentations = augmentations or []
        self.p = p
    
    def __call__(self, time_series):
        """
        Apply augmentations to a time series.
        
        Args:
            time_series (array): Time series to augment.
            
        Returns:
            array: Augmented time series.
        """
        augmented = time_series.copy()
        
        for augmentation in self.augmentations:
            if random.random() < self.p:
                augmented = augmentation(augmented)
        
        return augmented


def time_shift(time_series, max_shift_ratio=0.2):
    """
    Shift the time series along the time axis.
    
    Args:
        time_series (array): Time series to augment.
        max_shift_ratio (float): Maximum shift as a ratio of the time series length.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    max_shift = int(ts_length * max_shift_ratio)
    
    if max_shift == 0:
        return time_series
    
    shift = random.randint(-max_shift, max_shift)
    
    if shift > 0:
        # Shift right
        augmented = np.concatenate([time_series[-shift:], time_series[:-shift]])
    elif shift < 0:
        # Shift left
        shift = abs(shift)
        augmented = np.concatenate([time_series[shift:], time_series[:shift]])
    else:
        augmented = time_series.copy()
    
    return augmented


def time_stretch(time_series, min_ratio=0.8, max_ratio=1.2):
    """
    Stretch or compress the time series along the time axis.
    
    Args:
        time_series (array): Time series to augment.
        min_ratio (float): Minimum stretch ratio.
        max_ratio (float): Maximum stretch ratio.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    stretch_ratio = random.uniform(min_ratio, max_ratio)
    
    # Create new time points
    original_time = np.arange(ts_length)
    new_time = np.linspace(0, ts_length - 1, int(ts_length * stretch_ratio))
    
    # Interpolate
    interpolator = interp1d(original_time, time_series, kind='linear', bounds_error=False, fill_value='extrapolate')
    stretched = interpolator(new_time)
    
    # Ensure the output has the same length as the input
    if len(stretched) > ts_length:
        # Crop
        start = random.randint(0, len(stretched) - ts_length)
        augmented = stretched[start:start + ts_length]
    elif len(stretched) < ts_length:
        # Pad
        augmented = np.pad(stretched, (0, ts_length - len(stretched)), mode='wrap')
    else:
        augmented = stretched
    
    return augmented


def add_noise(time_series, noise_level=0.05):
    """
    Add Gaussian noise to the time series.
    
    Args:
        time_series (array): Time series to augment.
        noise_level (float): Standard deviation of the noise relative to the signal amplitude.
        
    Returns:
        array: Augmented time series.
    """
    ts_std = np.std(time_series)
    noise = np.random.normal(0, noise_level * ts_std, len(time_series))
    augmented = time_series + noise
    return augmented


def amplitude_scale(time_series, min_factor=0.7, max_factor=1.3):
    """
    Scale the amplitude of the time series.
    
    Args:
        time_series (array): Time series to augment.
        min_factor (float): Minimum scaling factor.
        max_factor (float): Maximum scaling factor.
        
    Returns:
        array: Augmented time series.
    """
    scale_factor = random.uniform(min_factor, max_factor)
    augmented = time_series * scale_factor
    return augmented


def random_crop(time_series, crop_ratio=0.8):
    """
    Randomly crop a segment of the time series and resize to original length.
    
    Args:
        time_series (array): Time series to augment.
        crop_ratio (float): Ratio of the original length to crop.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    crop_length = int(ts_length * crop_ratio)
    
    if crop_length >= ts_length:
        return time_series
    
    start = random.randint(0, ts_length - crop_length)
    cropped = time_series[start:start + crop_length]
    
    # Resize to original length using interpolation
    original_time = np.arange(crop_length)
    new_time = np.linspace(0, crop_length - 1, ts_length)
    
    interpolator = interp1d(original_time, cropped, kind='linear', bounds_error=False, fill_value='extrapolate')
    augmented = interpolator(new_time)
    
    return augmented


def frequency_mask(time_series, mask_ratio=0.1):
    """
    Apply a mask in the frequency domain.
    
    Args:
        time_series (array): Time series to augment.
        mask_ratio (float): Ratio of frequencies to mask.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    
    # FFT
    fft = np.fft.rfft(time_series)
    fft_mag = np.abs(fft)
    fft_phase = np.angle(fft)
    
    # Create mask
    mask_length = int(len(fft) * mask_ratio)
    mask_start = random.randint(0, len(fft) - mask_length)
    mask = np.ones(len(fft), dtype=bool)
    mask[mask_start:mask_start + mask_length] = False
    
    # Apply mask
    fft_mag_masked = fft_mag * mask
    fft_masked = fft_mag_masked * np.exp(1j * fft_phase)
    
    # IFFT
    augmented = np.fft.irfft(fft_masked, n=ts_length)
    
    return augmented


def baseline_wander(time_series, amplitude=0.1, frequency=0.05):
    """
    Add baseline wander to the time series.
    
    Args:
        time_series (array): Time series to augment.
        amplitude (float): Amplitude of the baseline wander relative to the signal amplitude.
        frequency (float): Frequency of the baseline wander relative to the sampling rate.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    ts_std = np.std(time_series)
    
    # Create baseline wander signal
    t = np.arange(ts_length)
    phase = random.uniform(0, 2 * np.pi)
    baseline = amplitude * ts_std * np.sin(2 * np.pi * frequency * t / ts_length + phase)
    
    # Add baseline wander
    augmented = time_series + baseline
    
    return augmented


def permutation(time_series, n_segments=4):
    """
    Randomly permute segments of the time series.
    
    Args:
        time_series (array): Time series to augment.
        n_segments (int): Number of segments to permute.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    segment_length = ts_length // n_segments
    
    # Split into segments
    segments = []
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < n_segments - 1 else ts_length
        segments.append(time_series[start:end])
    
    # Permute segments
    random.shuffle(segments)
    
    # Concatenate segments
    augmented = np.concatenate(segments)
    
    return augmented


def time_warping(time_series, n_knots=4, sigma=0.1):
    """
    Apply time warping to the time series.
    
    Args:
        time_series (array): Time series to augment.
        n_knots (int): Number of knots for the spline.
        sigma (float): Standard deviation of the knot displacement.
        
    Returns:
        array: Augmented time series.
    """
    ts_length = len(time_series)
    
    # Create knots
    knots = np.linspace(0, ts_length - 1, n_knots + 2)
    
    # Displace inner knots
    knots[1:-1] += np.random.normal(0, sigma * ts_length, n_knots)
    
    # Ensure knots are in ascending order
    knots = np.sort(knots)
    
    # Create warping function
    warping = interp1d(knots, knots, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Apply warping
    original_time = np.arange(ts_length)
    warped_time = warping(original_time)
    
    # Interpolate to get warped signal
    interpolator = interp1d(original_time, time_series, kind='linear', bounds_error=False, fill_value='extrapolate')
    augmented = interpolator(warped_time)
    
    return augmented


def get_augmenter(augmentation_types=None, p=0.5):
    """
    Get an augmenter with specified augmentation types.
    
    Args:
        augmentation_types (list): List of augmentation types to include.
        p (float): Probability of applying each augmentation.
        
    Returns:
        TimeSeriesAugmenter: Augmenter object.
    """
    augmentation_functions = {
        'time_shift': time_shift,
        'time_stretch': time_stretch,
        'add_noise': add_noise,
        'amplitude_scale': amplitude_scale,
        'random_crop': random_crop,
        'frequency_mask': frequency_mask,
        'baseline_wander': baseline_wander,
        'permutation': permutation,
        'time_warping': time_warping
    }
    
    if augmentation_types is None:
        augmentation_types = list(augmentation_functions.keys())
    
    augmentations = [augmentation_functions[aug_type] for aug_type in augmentation_types if aug_type in augmentation_functions]
    
    return TimeSeriesAugmenter(augmentations, p)


def augment_dataset(X, y, augmenter, augment_ratio=1.0, class_balance=False):
    """
    Augment a dataset.
    
    Args:
        X (list): List of time series.
        y (pandas.DataFrame): DataFrame with labels.
        augmenter (TimeSeriesAugmenter): Augmenter to use.
        augment_ratio (float): Ratio of augmented samples to original samples.
        class_balance (bool): Whether to balance classes through augmentation.
        
    Returns:
        tuple: (augmented_X, augmented_y)
    """
    augmented_X = X.copy()
    augmented_y = y.copy()
    
    if class_balance:
        # Count samples per class
        class_counts = y['label'].value_counts()
        max_count = class_counts.max()
        
        # Augment each class to have the same number of samples
        for class_label, count in class_counts.items():
            # Number of samples to add
            n_to_add = max_count - count
            
            if n_to_add <= 0:
                continue
            
            # Get indices of samples in this class
            indices = y.index[y['label'] == class_label].tolist()
            
            # Randomly select samples to augment (with replacement)
            selected_indices = np.random.choice(indices, size=n_to_add, replace=True)
            
            # Augment selected samples
            for idx in selected_indices:
                augmented_X.append(augmenter(X[idx]))
                augmented_y = augmented_y.append({'label': class_label}, ignore_index=True)
    else:
        # Augment a random subset of the data
        n_to_add = int(len(X) * augment_ratio)
        
        # Randomly select samples to augment (with replacement)
        selected_indices = np.random.choice(len(X), size=n_to_add, replace=True)
        
        # Augment selected samples
        for idx in selected_indices:
            augmented_X.append(augmenter(X[idx]))
            augmented_y = augmented_y.append({'label': y.iloc[idx]['label']}, ignore_index=True)
    
    return augmented_X, augmented_y


def main():
    """
    Main function to demonstrate augmentation.
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
        import numpy as np
        X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
        import pandas as pd
        y_train = pd.DataFrame({'label': [0]})
    else:
        # Load data
        try:
            X_train, y_train, _ = load_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating synthetic data for demonstration...")
            # Create synthetic data
            import numpy as np
            X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
            import pandas as pd
            y_train = pd.DataFrame({'label': [0]})
    
    if X_train is None:
        print("Could not load data. Creating synthetic data...")
        # Create synthetic data
        import numpy as np
        X_train = [np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)]
        import pandas as pd
        y_train = pd.DataFrame({'label': [0]})
    
    # Select a sample
    sample_idx = 0
    sample = X_train[sample_idx]
    label = y_train.iloc[sample_idx]['label']
    
    print(f"Original sample (class {label}):")
    
    # Create augmenter
    augmenter = get_augmenter()
    
    try:
        # Apply augmentations
        plt.figure(figsize=(15, 10))
        
        # Plot original
        plt.subplot(5, 2, 1)
        plt.plot(sample)
        plt.title("Original")
        
        # Plot augmentations
        augmentations = [
            ('Time Shift', time_shift),
            ('Time Stretch', time_stretch),
            ('Add Noise', add_noise),
            ('Amplitude Scale', amplitude_scale),
            ('Random Crop', random_crop),
            ('Frequency Mask', frequency_mask),
            ('Baseline Wander', baseline_wander),
            ('Permutation', permutation),
            ('Time Warping', time_warping)
        ]
        
        for i, (name, aug_func) in enumerate(augmentations):
            plt.subplot(5, 2, i + 2)
            augmented = aug_func(sample)
            plt.plot(augmented)
            plt.title(name)
        
        plt.tight_layout()
        plt.savefig("augmentation_examples.png")
        plt.close()
        
        print("Augmentation examples saved to 'augmentation_examples.png'")
    except Exception as e:
        print(f"Error applying augmentations: {e}")
        print("Please ensure all required packages are installed.")


if __name__ == "__main__":
    main() 