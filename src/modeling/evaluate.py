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
from reduction.reduce import read_custom_binary
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REDUCED_DIR = os.path.join(PROJECT_ROOT, 'reduced')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'augment', 'reduction'], default='baseline',
                        help='Which pipeline/model to use for evaluation')
    parser.add_argument('--reduced_file', default='train_25pct_kmeans.bin', help='Reduced binary file for reduction mode')
    parser.add_argument('--reduced_test_file', default='test_25pct_kmeans.bin', help='Reduced binary test set (optional)')
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
        rf_model_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_model.joblib')
        csv_prefix = 'base'
        X_train, y_train, X_test = load_dataset('data')
        idx = np.arange(len(X_train))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
        X_val = [X_train[i] for i in val_idx]
        y_val = y_train.iloc[val_idx].values
        X_eval = X_val
        y_eval = y_val
        X_test_eval = X_test
    elif mode == 'augment':
        rf_model_path = os.path.join(MODELS_DIR, 'rf_aug_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_aug_model.joblib')
        csv_prefix = 'augment'
        X_train, y_train, X_test = load_dataset('data')
        idx = np.arange(len(X_train))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
        X_val = [X_train[i] for i in val_idx]
        y_val = y_train.iloc[val_idx].values
        X_eval = X_val
        y_eval = y_val
        X_test_eval = X_test
    else:  # REDUCTION
        rf_model_path = os.path.join(MODELS_DIR, 'rf_reduced_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_reduced_model.joblib')
        csv_prefix = 'reduced'
        # Use reduced val/test data as in train.py
        reduced_path = os.path.join(REDUCED_DIR, args.reduced_file)
        X_reduced = read_custom_binary(reduced_path)
        label_path = reduced_path.replace('.bin', '.csv')
        y_reduced = pd.read_csv(label_path)
        idx = np.arange(len(X_reduced))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_reduced, random_state=42)
        X_eval = [X_reduced[i] for i in val_idx]
        y_eval = y_reduced.iloc[val_idx].values.ravel()

        # For test set, you need to generate reduced/compressed test data using same process.
        reduced_test_path = os.path.join(REDUCED_DIR, args.reduced_test_file)
        if os.path.exists(reduced_test_path):
            X_test_eval = read_custom_binary(reduced_test_path)
        else:
            print("Reduced test set binary not found. Using baseline test set as fallback.")
            _, _, X_test = load_dataset('data')
            X_test_eval = X_test  # fallback

    for name, model_path in zip(['rf', 'mlp'], [rf_model_path, mlp_model_path]):
        print(f"\nEvaluating {name.upper()} ({mode}) ...")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        model = joblib.load(model_path)
        y_val_pred = model.predict(X_eval)
        val_acc = accuracy_score(y_eval, y_val_pred)
        print(f"{name.upper()} validation accuracy: {val_acc:.4f}")
        print(f"{name.upper()} Classification report:\n{classification_report(y_eval, y_val_pred)}")
        plot_confusion(y_eval, y_val_pred, f"{name.upper()}-{mode}")

        y_test_pred = model.predict(X_test_eval)
        pred_df = pd.DataFrame({"label": y_test_pred})
        pred_path = os.path.join(MODELS_DIR, f"{name}_{csv_prefix}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved {pred_path}")

if __name__ == "__main__":
    main()
