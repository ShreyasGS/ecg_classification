import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loading import load_dataset
from augmentation.features import FeatureExtractor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'augment'], default='baseline',
                        help='Which pipeline/model to use for evaluation')
    return parser.parse_args()

def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    mode = args.mode

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    if mode == 'baseline':
        rf_model_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_model.joblib')
        csv_prefix = 'base'
    else:
        rf_model_path = os.path.join(MODELS_DIR, 'rf_aug_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_aug_model.joblib')
        csv_prefix = 'augment'

    # Load data
    X_train, y_train, X_test = load_dataset('data')
    idx = np.arange(len(X_train))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
    X_val = [X_train[i] for i in val_idx]
    y_val = y_train.iloc[val_idx].values

    # --- Explicit feature extraction if model pipeline does not contain it ---
    # Load a model and check if it has a "features" step; otherwise, extract features here
    fe = FeatureExtractor()
    # You *must* do this if you trained models on features, not signals!
    X_val_feat = fe.transform(X_val)
    X_test_feat = fe.transform(X_test)

    for name, model_path in zip(['rf', 'mlp'], [rf_model_path, mlp_model_path]):
        print(f"\nEvaluating {name.upper()} ({mode}) ...")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        model = joblib.load(model_path)
        try:
            # Try predicting with raw signal (works if pipeline includes feature extraction)
            y_val_pred = model.predict(X_val)
        except Exception:
            # Fallback: predict with explicit features
            y_val_pred = model.predict(X_val_feat)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"{name.upper()} validation accuracy: {val_acc:.4f}")
        print(f"{name.upper()} Classification report:\n{classification_report(y_val, y_val_pred)}")
        plot_confusion(y_val, y_val_pred, f"{name.upper()}-{mode}")

        try:
            y_test_pred = model.predict(X_test)
        except Exception:
            y_test_pred = model.predict(X_test_feat)
        pred_df = pd.DataFrame({"label": y_test_pred})
        pred_path = os.path.join(MODELS_DIR, f"{name}_{csv_prefix}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved {pred_path}")

if __name__ == "__main__":
    main()
