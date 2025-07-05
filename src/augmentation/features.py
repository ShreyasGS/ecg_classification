"""
features.py

ECG Feature Extraction Transformer
---------------------------------
Extracts various features from 1D ECG signals for use in ML models.
Can be plugged into a scikit-learn Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew, kurtosis
from scipy.signal import welch

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer to convert list of signals to DataFrame of features.
    """
    def __init__(self, fs=300):
        self.fs = fs  # Sampling rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feats = []
        for sig in X:
            arr = np.array(sig)
            ptp = float(arr.max() - arr.min())
            mad = float(np.mean(np.abs(arr - arr.mean())))
            sk = float(skew(arr))
            kt = float(kurtosis(arr))
            zcr = float(np.mean(np.diff(np.sign(arr)) != 0))
            energy = float(np.sum(arr**2))
            hist = np.histogram(arr, bins=50)[0] + 1
            probs = hist / hist.sum()
            entropy = float(-np.sum(probs * np.log2(probs)))
            # Frequency features
            f, Pxx = welch(arr, fs=self.fs, nperseg=1024)
            power_low = float(np.trapz(Pxx[(f>=0.5)&(f<4)], f[(f>=0.5)&(f<4)]))
            power_high = float(np.trapz(Pxx[(f>=20)&(f<50)], f[(f>=20)&(f<50)]))
            hf = np.trapz(Pxx[f > 40], f[f > 40])
            tot = np.trapz(Pxx, f)
            noise_ratio = float(hf / tot) if tot > 0 else 0.0
            feats.append({
                'ptp': ptp,
                'mad': mad,
                'skew': sk,
                'kurt': kt,
                'zcr': zcr,
                'energy': energy,
                'entropy': entropy,
                'power_low': power_low,
                'power_high': power_high,
                'noise_ratio': noise_ratio,
            })
        return pd.DataFrame(feats)

# ============== Optional: usage test ==============

if __name__ == "__main__":
    sig = np.sin(np.linspace(0, 10*np.pi, 9000)) + np.random.normal(0, 0.1, 9000)
    X = [sig] * 3
    fe = FeatureExtractor()
    feats_df = fe.fit_transform(X)
    print("Feature matrix:\n", feats_df.head())
