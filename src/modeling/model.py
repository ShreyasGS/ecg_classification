import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Make sure src/ is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loading import load_dataset
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'augment'], default='baseline',
                        help='Which pipeline to use: baseline or augmented')
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode

    if mode == 'baseline':
        from model import get_random_forest_pipeline, get_mlp_pipeline
        rf_pipeline = get_random_forest_pipeline()
        mlp_pipeline = get_mlp_pipeline()
        rf_model_path = '../../models/rf_model.joblib'
        mlp_model_path = '../../models/mlp_model.joblib'
        summary_path = '../../models/training_summary.csv'
    else:
        from model import get_rf_augmented_pipeline, get_mlp_augmented_pipeline
        rf_pipeline = get_rf_augmented_pipeline()
        mlp_pipeline = get_mlp_augmented_pipeline()
        rf_model_path = '../../models/rf_aug_model.joblib'
        mlp_model_path = '../../models/mlp_aug_model.joblib'
        summary_path = '../../models/training_aug_summary.csv'

    X_train, y_train, _ = load_dataset('data')
    idx = np.arange(len(X_train))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
    X_tr = [X_train[i] for i in train_idx]
    y_tr = y_train.iloc[train_idx].values
    X_val = [X_train[i] for i in val_idx]
    y_val = y_train.iloc[val_idx].values

    # Fit and evaluate Random Forest
    print(f"Training Random Forest ({mode})...")
    rf_pipeline.fit(X_tr, y_tr)
    val_rf_acc = rf_pipeline.score(X_val, y_val)
    print(f"RF val accuracy: {val_rf_acc:.4f}")
    joblib.dump(rf_pipeline, rf_model_path)
    print(f"Saved {rf_model_path}")

    # Fit and evaluate MLP
    print(f"Training MLP ({mode})...")
    mlp_pipeline.fit(X_tr, y_tr)
    val_mlp_acc = mlp_pipeline.score(X_val, y_val)
    print(f"MLP val accuracy: {val_mlp_acc:.4f}")
    joblib.dump(mlp_pipeline, mlp_model_path)
    print(f"Saved {mlp_model_path}")

    # Summary file
    summary = pd.DataFrame([
        {'model': 'rf', 'val_accuracy': val_rf_acc, 'mode': mode},
        {'model': 'mlp', 'val_accuracy': val_mlp_acc, 'mode': mode}
    ])
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
