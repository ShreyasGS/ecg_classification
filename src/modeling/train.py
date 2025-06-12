"""
train.py

Functions for training and evaluating ECG classification models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import seaborn as sns
import os
import sys
import time
from tqdm import tqdm

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.model import get_model, pad_collate
from data_loading import load_dataset


class ECGDataset(torch.utils.data.Dataset):
    """
    Dataset for ECG time series.
    """
    def __init__(self, time_series, labels=None, transform=None):
        """
        Args:
            time_series (list): List of time series.
            labels (pandas.DataFrame): DataFrame with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.time_series = time_series
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, idx):
        time_series = self.time_series[idx]
        
        if self.transform:
            time_series = self.transform(time_series)
        
        # Convert to tensor
        time_series = torch.tensor(time_series, dtype=torch.float32)
        
        if self.labels is not None:
            label = self.labels.iloc[idx]['label']
            return time_series, label
        else:
            return time_series


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, scheduler=None, early_stopping=5):
    """
    Train a model.
    
    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device (torch.device): Device to train on.
        num_epochs (int): Number of epochs.
        scheduler: Learning rate scheduler.
        early_stopping (int): Number of epochs to wait for improvement before stopping.
        
    Returns:
        tuple: (trained_model, history)
    """
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a model.
    
    Args:
        model (nn.Module): Model to evaluate.
        test_loader (DataLoader): Test data loader.
        criterion: Loss function.
        device (torch.device): Device to evaluate on.
        
    Returns:
        tuple: (loss, accuracy, predictions, true_labels)
    """
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, _ in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Save predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss /= test_total
    test_acc = test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, np.array(all_predictions), np.array(all_labels)


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


def plot_confusion_matrix(true_labels, predictions, class_names=None, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        true_labels (array): True labels.
        predictions (array): Predicted labels.
        class_names (list): Class names.
        save_path (str): Path to save the plot.
    """
    if class_names is None:
        class_names = ['Normal', 'AF', 'Other', 'Noisy']
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


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


def main():
    """
    Main function to train and evaluate a model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    X_train, y_train, X_test = load_dataset()
    
    if X_train is None:
        print("Could not load data. Exiting.")
        return
    
    # Create train/val split
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train['label']
    )
    
    # Create datasets
    train_dataset = ECGDataset(X_train_split, y_train_split)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    
    # Create model
    model = get_model('cnn')
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=20, scheduler=scheduler, early_stopping=5
    )
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')
    
    # Evaluate model
    loss, acc, predictions, true_labels = evaluate_model(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, save_path='confusion_matrix.png')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(true_labels, predictions, target_names=['Normal', 'AF', 'Other', 'Noisy']))
    
    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to model.pth')
    
    # Generate predictions for test set
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for inputs, _, _ in tqdm(test_loader, desc='Generating test predictions'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
    
    # Save test predictions
    save_predictions(test_predictions, 'base.csv')


if __name__ == "__main__":
    main() 