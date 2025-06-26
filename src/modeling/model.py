# src/modeling/model.py

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from augmentation.features import FeatureExtractor
from augmentation.augment import SignalAugmenter

def get_random_forest_pipeline():
    """Returns a pipeline for baseline: FeatureExtractor + RandomForest."""
    return Pipeline([
        ('features', FeatureExtractor()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def get_mlp_pipeline():
    """Returns a pipeline for baseline: FeatureExtractor + MLP."""
    return Pipeline([
        ('features', FeatureExtractor()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
    ])

def get_rf_augmented_pipeline():
    """Pipeline with augmentation for RandomForest."""
    return Pipeline([
        ('augment', SignalAugmenter(n_augments=1, random_state=42, train_mode=True)),
        ('features', FeatureExtractor()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def get_mlp_augmented_pipeline():
    """Pipeline with augmentation for MLP."""
    return Pipeline([
        ('augment', SignalAugmenter(n_augments=1, random_state=42, train_mode=True)),
        ('features', FeatureExtractor()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
    ])
