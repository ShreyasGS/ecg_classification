"""
model.py

Defines model architectures for ECG time series classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECGConv1D(nn.Module):
    """
    1D Convolutional Neural Network for ECG classification.
    
    Architecture:
    - Multiple convolutional layers with batch normalization and dropout
    - Global average pooling
    - Fully connected layers
    """
    def __init__(self, num_classes=4, input_channels=1, dropout_rate=0.2):
        super(ECGConv1D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ECGLSTM(nn.Module):
    """
    LSTM model for ECG classification.
    
    Architecture:
    - Bidirectional LSTM layers
    - Attention mechanism
    - Fully connected layers
    """
    def __init__(self, num_classes=4, input_size=1, hidden_size=128, num_layers=2, dropout_rate=0.2):
        super(ECGLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ECGResNet(nn.Module):
    """
    ResNet-like model for ECG classification.
    
    Architecture:
    - Multiple residual blocks with skip connections
    - Global average pooling
    - Fully connected layers
    """
    def __init__(self, num_classes=4, input_channels=1, dropout_rate=0.2):
        super(ECGResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with potential downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        out += self.skip(x)
        out = F.relu(out)
        
        return out


def pad_collate(batch):
    """
    Collate function for variable length sequences.
    
    Args:
        batch: List of (sequence, label) tuples.
        
    Returns:
        tuple: (padded_sequences, labels)
    """
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    # Get sequences and labels
    sequences, labels = zip(*batch)
    
    # Get sequence lengths
    lengths = [len(seq) for seq in sequences]
    max_length = max(lengths)
    
    # Pad sequences
    padded_sequences = torch.zeros(len(sequences), 1, max_length)
    for i, seq in enumerate(sequences):
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        padded_sequences[i, :, :len(seq)] = seq_tensor
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, labels, lengths


def get_model(model_name, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        model_name (str): Name of the model.
        **kwargs: Additional arguments for the model.
        
    Returns:
        nn.Module: Model instance.
    """
    models = {
        'cnn': ECGConv1D,
        'lstm': ECGLSTM,
        'resnet': ECGResNet
    }
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs) 