"""
evaluate.py

Loads saved models, evaluates on validation, and writes test predictions.
"""
import os
import sys
import joblib
import pandas as pd

# Ensure project src directory is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_loading import load_dataset, summarize_signal, noise_ratio


def build_features(X_raw):
    records = []
    for sig in X_raw:
        feats = summarize_signal(sig)
        feats['noise_ratio'] = noise_ratio(sig)
        records.append(feats)
    return pd.DataFrame(records)


def main(data_dir=None, model_dir=None):
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_dir = data_dir or os.path.join(project_root, 'data')
    model_dir = model_dir or os.path.join(project_root, 'models')

    # Load data
    X_val_raw, y_val, X_test_raw = load_dataset(data_dir)

    # Feature engineering
    X_val_feat = build_features(X_val_raw)
    X_test_feat = build_features(X_test_raw)

    for name in ['rf', 'mlp']:
        model_path = os.path.join(model_dir, f"{name}_model.joblib")
        model = joblib.load(model_path)
        val_acc = (model.predict(X_val_feat) == y_val).mean()
        print(f"{name} validation accuracy: {val_acc:.4f}")

        out_csv = os.path.join(model_dir, f"{name}_base.csv")
        pd.DataFrame(model.predict(X_test_feat), columns=['label']).to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")


if __name__ == '__main__':
    main()