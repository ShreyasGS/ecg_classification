"""
evaluate.py

Functions for evaluating ECG classification models and computing metrics.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from tqdm import tqdm
import os
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.model import get_model


def predict(model, data_loader, device):
    """
    Generate predictions for a dataset.
    
    Args:
        model (nn.Module): Model to use for predictions.
        data_loader (DataLoader): Data loader.
        device (torch.device): Device to use.
        
    Returns:
        tuple: (predictions, probabilities)
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, _, _ in tqdm(data_loader, desc='Generating predictions'):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Save predictions and probabilities
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)


def compute_metrics(true_labels, predictions, class_names=None):
    """
    Compute classification metrics.
    
    Args:
        true_labels (array): True labels.
        predictions (array): Predicted labels.
        class_names (list): Class names.
        
    Returns:
        dict: Dictionary of metrics.
    """
    if class_names is None:
        class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision_macro': precision_score(true_labels, predictions, average='macro'),
        'recall_macro': recall_score(true_labels, predictions, average='macro'),
        'f1_macro': f1_score(true_labels, predictions, average='macro'),
    }
    
    # Class-specific metrics
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_score(true_labels, predictions, average=None, labels=[i])[0]
        metrics[f'recall_{class_name}'] = recall_score(true_labels, predictions, average=None, labels=[i])[0]
        metrics[f'f1_{class_name}'] = f1_score(true_labels, predictions, average=None, labels=[i])[0]
    
    return metrics


def plot_confusion_matrix(true_labels, predictions, class_names=None, normalize=False, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        true_labels (array): True labels.
        predictions (array): Predicted labels.
        class_names (list): Class names.
        normalize (bool): Whether to normalize the confusion matrix.
        save_path (str): Path to save the plot.
    """
    if class_names is None:
        class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    cm = confusion_matrix(true_labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(true_labels, probabilities, class_names=None, save_path=None):
    """
    Plot ROC curves.
    
    Args:
        true_labels (array): True labels.
        probabilities (array): Predicted probabilities.
        class_names (list): Class names.
        save_path (str): Path to save the plot.
    """
    if class_names is None:
        class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    # One-hot encode true labels
    y_true = np.zeros((len(true_labels), len(class_names)))
    for i, label in enumerate(true_labels):
        y_true[i, label] = 1
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(true_labels, probabilities, class_names=None, save_path=None):
    """
    Plot precision-recall curves.
    
    Args:
        true_labels (array): True labels.
        probabilities (array): Predicted probabilities.
        class_names (list): Class names.
        save_path (str): Path to save the plot.
    """
    if class_names is None:
        class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    # One-hot encode true labels
    y_true = np.zeros((len(true_labels), len(class_names)))
    for i, label in enumerate(true_labels):
        y_true[i, label] = 1
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], probabilities[:, i])
        avg_precision = np.mean(precision)
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV file.
    
    Args:
        predictions (array): Predicted labels.
        output_path (str): Path to save the CSV file.
    """
    df = pd.DataFrame({'label': predictions})
    df.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')


def load_model(model_name, model_path, device, **kwargs):
    """
    Load a trained model.
    
    Args:
        model_name (str): Name of the model.
        model_path (str): Path to the model weights.
        device (torch.device): Device to load the model on.
        **kwargs: Additional arguments for the model.
        
    Returns:
        nn.Module: Loaded model.
    """
    model = get_model(model_name, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_and_save(model, data_loader, true_labels=None, device=None, output_path=None, class_names=None):
    """
    Evaluate a model and save results.
    
    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): Data loader.
        true_labels (array): True labels (optional).
        device (torch.device): Device to use.
        output_path (str): Path to save predictions.
        class_names (list): Class names.
        
    Returns:
        tuple: (predictions, probabilities, metrics)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if class_names is None:
        class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    # Generate predictions
    predictions, probabilities = predict(model, data_loader, device)
    
    # Compute metrics if true labels are provided
    metrics = None
    if true_labels is not None:
        metrics = compute_metrics(true_labels, predictions, class_names)
        
        # Print metrics
        print('\nEvaluation Metrics:')
        for metric_name, metric_value in metrics.items():
            print(f'{metric_name}: {metric_value:.4f}')
        
        # Plot confusion matrix
        plot_confusion_matrix(true_labels, predictions, class_names, normalize=False, save_path='confusion_matrix.png')
        plot_confusion_matrix(true_labels, predictions, class_names, normalize=True, save_path='normalized_confusion_matrix.png')
        
        # Plot ROC curve
        plot_roc_curve(true_labels, probabilities, class_names, save_path='roc_curve.png')
        
        # Plot precision-recall curve
        plot_precision_recall_curve(true_labels, probabilities, class_names, save_path='precision_recall_curve.png')
    
    # Save predictions
    if output_path:
        save_predictions(predictions, output_path)
    
    return predictions, probabilities, metrics


def main():
    """
    Main function to evaluate a model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model-name', type=str, default='cnn', help='Model name')
    parser.add_argument('--model-path', type=str, default='model.pth', help='Path to model weights')
    parser.add_argument('--output-path', type=str, default='predictions.csv', help='Path to save predictions')
    args = parser.parse_args()
    
    # Load data
    from data_loading import load_dataset
    X_train, y_train, X_test = load_dataset()
    
    if X_train is None:
        print("Could not load data. Exiting.")
        return
    
    # Create datasets and data loaders
    from torch.utils.data import DataLoader
    from modeling.train import ECGDataset
    from modeling.model import pad_collate
    
    # Create train/val split for evaluation
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train['label']
    )
    
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    
    # Load model
    model = load_model(args.model_name, args.model_path, device)
    
    # Evaluate on validation set
    print('\nEvaluating on validation set:')
    val_predictions, val_probabilities, val_metrics = evaluate_and_save(
        model, val_loader, y_val['label'].values, device, None
    )
    
    # Generate predictions for test set
    print('\nGenerating predictions for test set:')
    test_predictions, _, _ = evaluate_and_save(
        model, test_loader, None, device, args.output_path
    )


if __name__ == "__main__":
    main() 