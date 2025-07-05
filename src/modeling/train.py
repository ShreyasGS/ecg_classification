import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REDUCED_DIR = os.path.join(PROJECT_ROOT, 'reduced')
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loading import load_dataset
from reduction.reduce import read_custom_binary
from sklearn.model_selection import train_test_split
from augmentation.features import FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'augment', 'reduction'], default='baseline',
                        help='Which pipeline to use: baseline, augmented, or reduction')
    parser.add_argument('--reduced_file', default=f'train_25pct_kmeans.bin',
                        help='Path to reduced binary in /reduced (for reduction mode)')
    return parser.parse_args()

def main():
    args = parse_args()
    mode = args.mode

    if mode == 'baseline':
        from modeling.model import get_random_forest_pipeline, get_mlp_pipeline
        rf_pipeline = get_random_forest_pipeline()
        mlp_pipeline = get_mlp_pipeline()
        rf_model_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_model.joblib')
        summary_path = os.path.join(MODELS_DIR, 'training_summary.csv')
        # Standard pipeline with full data
        X_train, y_train, _ = load_dataset('data')
        idx = np.arange(len(X_train))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
        X_tr = [X_train[i] for i in train_idx]
        y_tr = y_train.iloc[train_idx].values
        X_val = [X_train[i] for i in val_idx]
        y_val = y_train.iloc[val_idx].values
    elif mode == 'augment':
        from modeling.model import get_rf_augmented_pipeline, get_mlp_augmented_pipeline
        rf_pipeline = get_rf_augmented_pipeline()
        mlp_pipeline = get_mlp_augmented_pipeline()
        rf_model_path = os.path.join(MODELS_DIR, 'rf_aug_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_aug_model.joblib')
        summary_path = os.path.join(MODELS_DIR, 'training_aug_summary.csv')
        # Augment pipeline with full data
        X_train, y_train, _ = load_dataset('data')
        idx = np.arange(len(X_train))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_train, random_state=42)
        X_tr = [X_train[i] for i in train_idx]
        y_tr = y_train.iloc[train_idx].values
        X_val = [X_train[i] for i in val_idx]
        y_val = y_train.iloc[val_idx].values
    else:  # REDUCTION
        from modeling.model import get_rf_reduced_pipeline, get_mlp_reduced_pipeline
        rf_pipeline = get_rf_reduced_pipeline()
        mlp_pipeline = get_mlp_reduced_pipeline()
        rf_model_path = os.path.join(MODELS_DIR, 'rf_reduced_model.joblib')
        mlp_model_path = os.path.join(MODELS_DIR, 'mlp_reduced_model.joblib')
        summary_path = os.path.join(MODELS_DIR, 'training_reduced_summary.csv')

        # Load reduced data: expects a bin file from /reduced/
        reduced_path = os.path.join(REDUCED_DIR, args.reduced_file)
        X_reduced = read_custom_binary(reduced_path)    # returns list of np arrays
        # Load labels: you must also save y_sub for each reduction set in reduction workflow as .csv (same name)
        label_path = reduced_path.replace('.bin', '.csv')
        y_reduced = pd.read_csv(label_path)
        # Split (use e.g., 80/20)
        idx = np.arange(len(X_reduced))
        train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=y_reduced, random_state=42)
        X_tr = [X_reduced[i] for i in train_idx]
        y_tr = y_reduced.iloc[train_idx].values.ravel()
        X_val = [X_reduced[i] for i in val_idx]
        y_val = y_reduced.iloc[val_idx].values.ravel()
        
        # For reduced data, we need to extract features before training
        feature_extractor = FeatureExtractor()
        X_tr = feature_extractor.transform(X_tr)
        X_val = feature_extractor.transform(X_val)

    # --- Training and summary as before ---
    print(f"Training Random Forest ({mode})...")
    rf_pipeline.fit(X_tr, y_tr)
    val_rf_acc = rf_pipeline.score(X_val, y_val)
    print(f"RF val accuracy: {val_rf_acc:.4f}")
    joblib.dump(rf_pipeline, rf_model_path)
    print(f"Saved {rf_model_path}")

    print(f"Training MLP ({mode})...")
    mlp_pipeline.fit(X_tr, y_tr)
    val_mlp_acc = mlp_pipeline.score(X_val, y_val)
    print(f"MLP val accuracy: {val_mlp_acc:.4f}")
    joblib.dump(mlp_pipeline, mlp_model_path)
    print(f"Saved {mlp_model_path}")

    # Summary
    summary = pd.DataFrame([
        {'model': 'rf', 'val_accuracy': val_rf_acc, 'mode': mode},
        {'model': 'mlp', 'val_accuracy': val_mlp_acc, 'mode': mode}
    ])
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
