# AMLS SoSe 2025: ECG Time Series Classification

This project provides a comprehensive pipeline for ECG (electrocardiogram) time series classification. It includes modules for data loading, feature extraction, data augmentation, dimensionality reduction, and deep learning model training and evaluation. The codebase is designed for both research and educational purposes, supporting reproducible experiments and extensible workflows for ECG signal analysis.

## Structure

- `src/`: Source code for data loading, exploration, modeling, augmentation, and reduction.
- `notebooks/`: Jupyter notebooks for data exploration and prototyping.
- `data/`: Place raw and processed data here (not tracked by git).
- `tests/`: Unit tests.

## Module Descriptions

- **src/data_loading.py**  
  Functions for loading and preprocessing ECG time series data from binary and zipped files, as well as loading label CSVs. Provides a unified interface for loading the full dataset for training and testing.

- **src/augmentation/features.py**  
  Extracts a wide range of features from ECG signals, including statistical, frequency-domain, wavelet, and ECG-specific (e.g., heart rate, HRV) features. Includes robust fallback logic if specialized libraries are missing.

- **src/augmentation/augment.py**  
  Implements various data augmentation techniques for time series, such as time shifting, stretching, noise addition, amplitude scaling, cropping, frequency masking, and more. Useful for increasing data diversity during model training.

- **src/reduction/reduce.py**  
  Contains methods for reducing the dimensionality or size of ECG time series data, including downsampling, piecewise approximation, wavelet and Fourier compression, quantization, and more. Also includes utilities for custom binary formats.

- **src/reduction/metrics.py**  
  Provides metrics for evaluating the quality of data reduction techniques, such as MAE, RMSE, PRD, SNR, and compression ratio. Also includes plotting and CSV export utilities for comparing methods.

- **src/modeling/model.py**  
  Defines neural network architectures (e.g., CNN, LSTM, ResNet) for ECG time series classification.

- **src/modeling/train.py**  
  Implements the training pipeline for deep learning models, including data loading, augmentation, training loops, and checkpointing.

- **src/modeling/evaluate.py**  
  Contains evaluation routines for trained models, including metrics calculation and reporting.

- **src/exploration/explore_data.py**  
  Provides tools for exploratory data analysis and visualization of ECG signals and their properties.

## Setup

1. Clone the repo.
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Add data files to the `data/` directory.

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
   - Reduces the dimensionality or size of the ECG data.

6. **Metrics Evaluation**
   - `python -m src.reduction.metrics`
   - Evaluates the quality of reduction techniques using various metrics.

7. **Model Training**
   - `python -m src.modeling.train`
   - Trains deep learning models on the (optionally augmented/reduced) data.

8. **Model Evaluation**
   - `python -m src.modeling.evaluate`
   - Evaluates the trained models and reports performance metrics.

## Generating Submission CSV Files

To generate the required test prediction CSV files (`base.csv`, `augment.csv`, `reduced.csv`), run the evaluation script for each of your trained models:

```
python -m src.modeling.evaluate --model-name <model_name> --model-path <model_weights.pth> --output-path <output_csv>
```

Replace `<model_name>` and `<model_weights.pth>` with your actual model type and checkpoint file for each case:

- **Base model:**
  ```
  python -m src.modeling.evaluate --model-name cnn --model-path base_model.pth --output-path base.csv
  ```
- **Augmented model:**
  ```
  python -m src.modeling.evaluate --model-name cnn --model-path augment_model.pth --output-path augment.csv
  ```
- **Reduced model:**
  ```
  python -m src.modeling.evaluate --model-name cnn --model-path reduced_model.pth --output-path reduced.csv
  ```

Each CSV will be saved in the root directory and should match the format of `y_train.csv` (columns: `id`, `label`).



