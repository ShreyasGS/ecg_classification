"""
explore_data.py

Functions for exploring and visualizing ECG time series data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import os

# Add parent directory to path to import data_loading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loading import load_dataset


def compute_statistics(time_series_list):
    """
    Compute basic statistics for a list of time series.
    
    Args:
        time_series_list (list): List of time series data.
        
    Returns:
        dict: Dictionary of statistics.
    """
    lengths = [len(ts) for ts in time_series_list]
    
    # Convert to numpy arrays for computation
    all_values = np.concatenate(time_series_list)
    
    stats = {
        'num_samples': len(time_series_list),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_value': np.mean(all_values),
        'std_value': np.std(all_values),
        'min_value': np.min(all_values),
        'max_value': np.max(all_values),
        'median_value': np.median(all_values),
        'q1_value': np.percentile(all_values, 25),
        'q3_value': np.percentile(all_values, 75),
    }
    
    return stats


def compute_class_statistics(X, y):
    """
    Compute statistics for each class.
    
    Args:
        X (list): List of time series.
        y (pandas.DataFrame): DataFrame with labels.
        
    Returns:
        dict: Dictionary of class statistics.
    """
    class_stats = {}
    
    for class_label in sorted(y['label'].unique()):
        indices = y.index[y['label'] == class_label].tolist()
        class_data = [X[i] for i in indices]
        
        stats = compute_statistics(class_data)
        class_stats[f'Class {class_label}'] = stats
    
    return class_stats


def plot_class_distribution(y):
    """
    Plot the class distribution.
    
    Args:
        y (pandas.DataFrame): DataFrame with labels.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=y)
    plt.title('Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2, 3], ['Normal', 'AF', 'Other', 'Noisy'])
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()


def plot_sample_time_series(X, y, num_samples=3):
    """
    Plot sample time series from each class.
    
    Args:
        X (list): List of time series.
        y (pandas.DataFrame): DataFrame with labels.
        num_samples (int): Number of samples to plot per class.
    """
    class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    fig, axes = plt.subplots(4, num_samples, figsize=(15, 10))
    
    for class_idx, class_label in enumerate(sorted(y['label'].unique())):
        indices = y.index[y['label'] == class_label].tolist()
        
        # Select random samples
        if len(indices) >= num_samples:
            sample_indices = np.random.choice(indices, num_samples, replace=False)
        else:
            sample_indices = indices
        
        for i, sample_idx in enumerate(sample_indices):
            if i >= num_samples:
                break
                
            time_series = X[sample_idx]
            
            # Plot
            ax = axes[class_idx, i]
            ax.plot(time_series)
            
            if i == 0:
                ax.set_ylabel(f'Class {class_label}\n({class_names[class_label]})')
            
            if class_idx == 0:
                ax.set_title(f'Sample {i+1}')
            
            ax.set_xticks([])
            
            # Add length information
            ax.text(0.5, 0.02, f'Length: {len(time_series)}', 
                    transform=ax.transAxes, ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('sample_time_series.png')
    plt.close()


def create_validation_split(X, y, test_size=0.2, random_state=42):
    """
    Create a stratified train/validation split.
    
    Args:
        X (list): List of time series.
        y (pandas.DataFrame): DataFrame with labels.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random state for reproducibility.
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    # Convert to numpy arrays for sklearn
    X_array = np.array(X, dtype=object)
    y_array = y['label'].values
    
    # Create stratified split
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        np.arange(len(X)), y_array, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_array
    )
    
    # Get the actual time series
    X_train = [X[i] for i in X_train_idx]
    X_val = [X[i] for i in X_val_idx]
    
    # Convert back to DataFrames
    y_train_df = pd.DataFrame({'label': y_train})
    y_val_df = pd.DataFrame({'label': y_val})
    
    return X_train, X_val, y_train_df, y_val_df


def create_kfold_splits(X, y, n_splits=5, random_state=42):
    """
    Create stratified k-fold splits.
    
    Args:
        X (list): List of time series.
        y (pandas.DataFrame): DataFrame with labels.
        n_splits (int): Number of folds.
        random_state (int): Random state for reproducibility.
        
    Returns:
        list: List of (X_train, X_val, y_train, y_val) tuples.
    """
    # Convert to numpy arrays for sklearn
    X_array = np.array(X, dtype=object)
    y_array = y['label'].values
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in skf.split(X_array, y_array):
        # Get the actual time series
        X_train = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        
        # Convert back to DataFrames
        y_train_df = pd.DataFrame({'label': y_array[train_idx]})
        y_val_df = pd.DataFrame({'label': y_array[val_idx]})
        
        splits.append((X_train, X_val, y_train_df, y_val_df))
    
    return splits


def main():
    """
    Main function to explore the dataset.
    """
    # Load data
    X_train, y_train, X_test = load_dataset()
    
    if X_train is None:
        print("Could not load data. Exiting.")
        return
    
    # Compute overall statistics
    print("Computing overall statistics...")
    stats = compute_statistics(X_train)
    print("Overall statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Compute class statistics
    print("\nComputing class statistics...")
    class_stats = compute_class_statistics(X_train, y_train)
    for class_name, stats in class_stats.items():
        print(f"\n{class_name} statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Plot class distribution
    print("\nPlotting class distribution...")
    plot_class_distribution(y_train)
    print("Class distribution plot saved as 'class_distribution.png'")
    
    # Plot sample time series
    print("\nPlotting sample time series...")
    plot_sample_time_series(X_train, y_train)
    print("Sample time series plot saved as 'sample_time_series.png'")
    
    # Create validation split
    print("\nCreating validation split...")
    X_train_split, X_val, y_train_split, y_val = create_validation_split(X_train, y_train)
    print(f"Train set size: {len(X_train_split)}, Validation set size: {len(X_val)}")
    
    # Check class distribution in splits
    print("\nClass distribution in train set:")
    print(y_train_split['label'].value_counts())
    print("\nClass distribution in validation set:")
    print(y_val['label'].value_counts())


if __name__ == "__main__":
    main() 