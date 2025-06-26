import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loading import load_dataset

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

    if mode == 'baseline':
        rf_model_path = '../../models/rf_model.joblib'
        mlp_model_path = '../../models/mlp_model.joblib'
        csv_prefix = 'base'
    else:
        rf_model_path = '../../models/rf_aug_model.joblib'
        mlp_model_path = '../../models/mlp_aug_model.joblib'
        csv_prefix = 'augment'

    # Load data
    X_train, y_train, X_test = load_dataset('data')
    idx = np.arange(len(X_train))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
    X_val = [X_train[i] for i in val_idx]
    y_val = y_train.iloc[val_idx].values

    for name, model_path in zip(['rf', 'mlp'], [rf_model_path, mlp_model_path]):
        print(f"\nEvaluating {name.upper()} ({mode}) ...")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        model = joblib.load(model_path)
        # ONLY pass the raw signals, let the pipeline do feature extraction/augmentation as needed
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"{name.upper()} validation accuracy: {val_acc:.4f}")
        print(f"{name.upper()} Classification report:\n{classification_report(y_val, y_val_pred)}")
        plot_confusion(y_val, y_val_pred, f"{name.upper()}-{mode}")

        # Test prediction
        y_test_pred = model.predict(X_test)
        pred_df = pd.DataFrame({"label": y_test_pred})
        pred_path = os.path.join('../../models', f"{name}_{csv_prefix}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved {pred_path}")

if __name__ == "__main__":
    main()
