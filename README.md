# AMLS SoSe 2025: ECG Time Series Classification

This project provides a comprehensive pipeline for ECG (electrocardiogram) time series classification. It includes modules for data loading, feature extraction, data augmentation, dimensionality reduction, and machine learning model training and evaluation. The codebase is designed for both research and educational purposes, supporting reproducible experiments and extensible workflows for ECG signal analysis.

## Structure

- `src/`: Source code for data loading, exploration, modeling, augmentation, and reduction.
- `notebooks/`: Jupyter notebooks for data exploration and prototyping.
- `data/`: Place raw and processed data here (not tracked by git).
- `models/`: Trained model files and evaluation results.
- `reduced/`: Output from data reduction techniques.

## Module Descriptions

- **src/data_loading.py**  
  Functions for loading and preprocessing ECG time series data from binary and zipped files, as well as loading label CSVs. Provides a unified interface for loading the full dataset for training and testing.

- **src/augmentation/features.py**  
  Extracts a wide range of features from ECG signals, including statistical, frequency-domain, wavelet, and ECG-specific features. Implements a scikit-learn compatible `FeatureExtractor` transformer for use in ML pipelines.

- **src/augmentation/augment.py**  
  Implements various data augmentation techniques for time series, such as time shifting, amplitude scaling, noise addition, and more. Includes a scikit-learn compatible `SignalAugmenter` transformer.

- **src/reduction/reduce.py**  
  Contains methods for reducing the dimensionality or size of ECG time series data, including downsampling, piecewise approximation, wavelet and Fourier compression, quantization, and more. Also includes utilities for custom binary formats and coreset selection.

- **src/reduction/metrics.py**  
  Provides metrics for evaluating the quality of data reduction techniques, such as MAE, RMSE, PRD, SNR, and compression ratio. Also includes plotting and CSV export utilities for comparing methods.

- **src/modeling/model.py**  
  Defines scikit-learn compatible pipelines for ECG classification, including feature extraction, augmentation, and model training.

- **src/modeling/train.py**  
  Implements the training pipeline for machine learning models with different modes: baseline, augmentation, and reduction.

- **src/modeling/evaluate.py**  
  Contains evaluation routines for trained models, including metrics calculation, reporting, and generating submission files.

- **src/exploration/explore_data.py**  
  Provides tools for exploratory data analysis and visualization of ECG signals and their properties.

## Setup

1. Clone the repo.
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Add data files to the `data/` directory:
   - `X_train.zip`: Training ECG time series data
   - `X_test.zip`: Test ECG time series data
   - `y_train.csv`: Training labels

## Tasks

- Data exploration and visualization
- Model training and evaluation
- Data augmentation and feature engineering
- Data reduction and analysis

## How to Run (Recommended Order)

1. **Data Loading**
   - `python src/data_loading.py`
   - Loads and checks your ECG data and labels.

2. **Exploratory Data Analysis**
   - `python -m src.exploration.explore_data`
   - Visualizes and explores the raw ECG signals and label distributions.

3. **Feature Extraction**
   - `python -m src.augmentation.features`
   - Extracts statistical, frequency, wavelet, and ECG-specific features.

4. **Data Augmentation**
   - `python -m src.augmentation.augment`
   - Applies augmentation techniques to increase data diversity.

5. **Data Reduction**
   - `python -m src.reduction.reduce`
   - Reduces the dimensionality or size of the ECG data using various techniques.
   - Creates binary files and corresponding label CSVs in the `reduced/` directory.

6. **Model Training**
   - Train baseline model:
     ```
     python -m src.modeling.train --mode baseline
     ```
   - Train model with augmented data:
     ```
     python -m src.modeling.train --mode augment
     ```
   - Train model with reduced data:
     ```
     python -m src.modeling.train --mode reduction --reduced_file train_25pct_kmeans.bin
     ```
   - Models are saved to the `models/` directory.

7. **Model Evaluation and Generating Submission Files**
   - Evaluate baseline model and generate `base.csv`:
     ```
     python -m src.modeling.evaluate --mode baseline --model-path models/rf_model.joblib --output-path base.csv
     ```
   - Evaluate augmented model and generate `augment.csv`:
     ```
     python -m src.modeling.evaluate --mode augment --model-path models/rf_aug_model.joblib --output-path augment.csv
     ```
   - Evaluate reduced model and generate `reduced.csv`:
     ```
     python -m src.modeling.evaluate --mode reduction --model-path models/rf_reduced_model.joblib --output-path reduced.csv
     ```

## Submission Files

The evaluation script generates three CSV files required for submission:

1. **base.csv**: Predictions from the baseline model
2. **augment.csv**: Predictions from the model trained with augmented data
3. **reduced.csv**: Predictions from the model trained with reduced data

Each CSV file contains two columns:
- `id`: Row identifier (0-indexed)
- `label`: Predicted class (0: Normal, 1: AF, 2: Other, 3: Noisy)

## Results

Our models achieved the following validation accuracies:
- Base model: 61.97%
- Augmented model: 60.92%
- Reduced model: 62.14%

The reduced model (trained on just 25% of the data using k-means selection) performed slightly better than the base model, demonstrating the effectiveness of our data reduction techniques in preserving important signal information while reducing noise.



