"""
data_loading.py

Functions for loading and preprocessing ECG time series data.
"""
import struct
import zipfile
import numpy as np
import pandas as pd
import os


def read_zip_binary(path):
    """
    Read binary data from a zipped file.
    
    Args:
        path (str): Path to the zip file.
        
    Returns:
        list: List of ECG time series.
    """
    ragged_array = []
    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = path.split("/")[-1].split(".")[0]
        with zf.open(f'{inner_path}.bin', 'r') as r:
            read_binary_from(ragged_array, r)
    return ragged_array


def read_binary(path):
    """
    Read binary data from an unzipped file.
    
    Args:
        path (str): Path to the binary file.
        
    Returns:
        list: List of ECG time series.
    """
    ragged_array = []
    with open(path, "rb") as r:
        read_binary_from(ragged_array, r)
    return ragged_array


def read_binary_from(ragged_array, r):
    """
    Helper function to read binary data from a file object.
    
    Args:
        ragged_array (list): List to append time series to.
        r (file): File object to read from.
    """
    while True:
        size_bytes = r.read(4)
        if not size_bytes:
            break
        sub_array_size = struct.unpack('i', size_bytes)[0]
        sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2)))
        ragged_array.append(sub_array)


def load_labels(path):
    """
    Load labels from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        pandas.DataFrame: DataFrame with labels.
    """
    return pd.read_csv(path)


def load_dataset(data_dir="data"):
    """
    Load the complete dataset (train and test).
    
    Args:
        data_dir (str): Directory containing the data files.
        
    Returns:
        tuple: (X_train, y_train, X_test)
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        print("Please place the data files in this directory and run again.")
        return None, None, None
    
    # Paths to data files
    train_data_path = os.path.join(data_dir, "X_train.zip")
    train_labels_path = os.path.join(data_dir, "y_train.csv")
    test_data_path = os.path.join(data_dir, "X_test.zip")
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [train_data_path, train_labels_path, test_data_path]):
        print("Missing data files. Please ensure all required files are in the data directory.")
        return None, None, None
    
    # Load data
    print("Loading training data...")
    X_train = read_zip_binary(train_data_path)
    
    print("Loading training labels...")
    y_train = load_labels(train_labels_path)
    
    print("Loading test data...")
    X_test = read_zip_binary(test_data_path)
    
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples.")
    return X_train, y_train, X_test


if __name__ == "__main__":
    # Example usage
    X_train, y_train, X_test = load_dataset()
    if X_train is not None:
        print(f"First training sample length: {len(X_train[0])}")
        print(f"Label distribution:\n{y_train['label'].value_counts()}") 