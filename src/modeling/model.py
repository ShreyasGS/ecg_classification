"""
model.py

Defines two scikit-learn Pipelines for ECG classification:
 - RandomForestClassifier
 - MLPClassifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_random_forest_pipeline() -> Pipeline:
    """
    Return a Pipeline with:
      - StandardScaler()
      - RandomForestClassifier(random_state=42)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])


def get_mlp_pipeline() -> Pipeline:
    """
    Return a Pipeline with:
      - StandardScaler()
      - MLPClassifier(max_iter=200, random_state=42)
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=200, random_state=42))
    ])
