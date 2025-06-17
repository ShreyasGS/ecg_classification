"""
train.py

Loads ECG data, builds engineered features, trains two models with GridSearchCV,
then saves estimators and a summary CSV.
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

# Add project src to path to import data_loading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from data_loading import load_dataset, summarize_signal, noise_ratio
from model import get_random_forest_pipeline, get_mlp_pipeline

# Hyperparameter grids
PARAM_GRIDS = {
    'rf': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 20]
    },
    'mlp': {
        'clf__hidden_layer_sizes': [(50,), (100,)],
        'clf__alpha': [0.0001, 0.001]
    }
}


def build_features(X_raw, y):
    records = []
    for sig, lbl in zip(X_raw, y):
        feats = summarize_signal(sig)
        feats['noise_ratio'] = noise_ratio(sig)
        feats['label'] = lbl
        records.append(feats)
    df = pd.DataFrame(records)
    return df.drop(columns=['label']), df['label']


def main(data_dir=None, model_dir=None):
    # Derive project paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_dir = data_dir or os.path.join(project_root, 'data')
    model_dir = model_dir or os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    X_train, y_train, _ = load_dataset(data_dir)

    # Feature engineering
    X_feat, y_feat = build_features(X_train, y_train)

    # Train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_feat, y_feat, test_size=0.2, stratify=y_feat, random_state=42
    )

    results = {}
    for name, pipeline_fn in [('rf', get_random_forest_pipeline), ('mlp', get_mlp_pipeline)]:
        print(f"Training {name}...")
        pipeline = pipeline_fn()
        grid = GridSearchCV(pipeline, PARAM_GRIDS[name], cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_
        val_score = best.score(X_val, y_val)
        results[name] = {
            'best_params': grid.best_params_,
            'train_score': grid.best_score_,
            'val_score': val_score
        }
        # Save model
        joblib.dump(best, os.path.join(model_dir, f"{name}_model.joblib"))
        print(f"Saved {name}_model.joblib with val_score={val_score:.4f}")

    # Save summary
    pd.DataFrame(results).T.to_csv(os.path.join(model_dir, 'training_summary.csv'))
    print(f"✅ Training complete. Summary at {model_dir}/training_summary.csv")


if __name__ == '__main__':
    main()
