"""
augment.py

ECG Data Augmentation Utilities for ML Pipelines
------------------------------------------------
- Atomic augmentations: time shifting, amplitude scaling, additive noise, random cropping.
- Composable master augmentation function.
- Scikit-learn compatible transformer: SignalAugmenter.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ============== Atomic augmentation functions ==============

def time_shift(sig, max_shift=100):
    """Randomly shift signal left/right by up to max_shift; pads with zeros."""
    arr = np.asarray(sig)
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.concatenate([np.zeros(shift), arr[:-shift]])
    elif shift < 0:
        return np.concatenate([arr[-shift:], np.zeros(-shift)])
    else:
        return arr

def amplitude_scale(sig, scale_range=(0.8, 1.2)):
    """Multiply amplitude by a random factor (simulate electrode variation)."""
    arr = np.asarray(sig)
    scale = np.random.uniform(*scale_range)
    return arr * scale

def add_noise(sig, noise_level=0.01):
    """Add Gaussian noise (level relative to signal std)."""
    arr = np.asarray(sig)
    noise = np.random.normal(0, noise_level * np.std(arr), size=arr.shape)
    return arr + noise

def random_crop(sig, crop_size):
    """Randomly crop the signal to a given crop_size, padding if necessary."""
    arr = np.asarray(sig)
    if len(arr) <= crop_size:
        out = np.zeros(crop_size)
        out[:len(arr)] = arr
        return out
    start = np.random.randint(0, len(arr) - crop_size)
    return arr[start : start + crop_size]

# ============== Compose multiple augmentations ==============

def augment_signal(sig):
    """
    Randomly applies several augmentations in sequence. Tune for your task.
    """
    aug_sig = np.asarray(sig).copy()
    if np.random.rand() < 0.5:
        aug_sig = time_shift(aug_sig)
    if np.random.rand() < 0.5:
        aug_sig = amplitude_scale(aug_sig)
    if np.random.rand() < 0.5:
        aug_sig = add_noise(aug_sig)
    # if np.random.rand() < 0.3:
    #     aug_sig = random_crop(aug_sig, crop_size=int(0.9 * len(aug_sig)))
    return aug_sig

# ============== Sklearn transformer ==============

class SignalAugmenter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for data augmentation.
    - Only use on training set! Set train_mode=False for val/test.
    """
    def __init__(self, n_augments=1, random_state=None, train_mode=True):
        self.n_augments = n_augments
        self.random_state = random_state
        self.train_mode = train_mode

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.train_mode:
            return X
        np.random.seed(self.random_state)
        X_aug = []
        for sig in X:
            sig_aug = np.asarray(sig).copy()
            for _ in range(self.n_augments):
                sig_aug = augment_signal(sig_aug)
            X_aug.append(sig_aug)
        return X_aug

# ============== Optional: usage test ==============

if __name__ == "__main__":
    sig = np.sin(np.linspace(0, 10*np.pi, 9000)) + np.random.normal(0, 0.1, 9000)
    aug = augment_signal(sig)
    print("Signal augmented! std before:", np.std(sig), "after:", np.std(aug))
    # Batch usage:
    augmenter = SignalAugmenter(n_augments=1, random_state=42)
    batch = [sig]*5
    batch_aug = augmenter.fit_transform(batch)
    print("Batch augmentation works!", np.shape(batch_aug))
