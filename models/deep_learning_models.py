"""
Advanced Deep Learning Models for Medical AI
Implements: CNN, RNN, LSTM, BiLSTM, Attention, Transformer
For medical text, images, and time-series analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# ==================== CNN MODELS ====================

class MedicalImageCNN(nn.Module):
    """
    Convolutional Neural Network for Medical Image Analysis
    Use cases: X-ray, CT, MRI classification
    """
    
    def __init__(self, num_classes: int = 14, input_channels: int = 3):
        """
        Args:
            num_classes: Number of disease classes (e.g., 14 for ChestX-ray14)
            input_channels: 1 for grayscale, 3 for RGB
        """
        super(MedicalImageCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Assuming 224x224 input
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper CNN"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MedicalResNet(nn.Module):
    """
    ResNet-style CNN for medical imaging
    Better accuracy than basic CNN
    """
    
    def __init__(self, num_classes: int = 14, input_channels: int = 3):
        super(MedicalResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ==================== RNN/LSTM MODELS ====================

class MedicalTextRNN(nn.Module):
    """
    Recurrent Neural Network for medical text analysis
    Use cases: Clinical notes, patient history, symptoms
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 10, num_layers: int = 2):
        super(MedicalTextRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # RNN
        output, hidden = self.rnn(embedded)  # output: (batch, seq_len, hidden_dim)
        
        # Use last hidden state
        last_hidden = output[:, -1, :]  # (batch, hidden_dim)
        
        # Classification
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


class MedicalTextLSTM(nn.Module):
    """
    LSTM for medical text - Better than RNN for long sequences
    Use cases: Long clinical notes, medical literature
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 10, num_layers: int = 2):
        super(MedicalTextLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.3, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Classification
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


class MedicalBiLSTM(nn.Module):
    """
    Bidirectional LSTM - Best for medical text understanding
    Reads text forward and backward for better context
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 10, num_layers: int = 2):
        super(MedicalBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=0.3, bidirectional=True)
        
        # *2 because bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        
        # Concatenate forward and backward hidden states
        last_hidden = lstm_out[:, -1, :]
        
        # Classification
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


class MedicalLSTMWithAttention(nn.Module):
    """
    LSTM with Attention Mechanism
    Focuses on important parts of medical text
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 10):
        super(MedicalLSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classification
        out = self.dropout(context)
        out = self.fc(out)
        
        return out


# ==================== TIME SERIES MODELS ====================

class VitalSignsLSTM(nn.Module):
    """
    LSTM for time-series vital signs prediction
    Use cases: Heart rate, blood pressure, glucose trends
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 2, output_dim: int = 1):
        super(VitalSignsLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        last_out = lstm_out[:, -1, :]
        
        # Prediction
        out = self.fc(last_out)
        
        return out


# ==================== MULTIMODAL MODELS ====================

class MultimodalMedicalNet(nn.Module):
    """
    Combines image (CNN) and text (LSTM) for diagnosis
    Use cases: Radiology report + X-ray image
    """
    
    def __init__(self, num_classes: int = 10, vocab_size: int = 10000):
        super(MultimodalMedicalNet, self).__init__()
        
        # Image branch (CNN)
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Text branch (LSTM)
        self.text_embedding = nn.Embedding(vocab_size, 128)
        self.text_lstm = nn.LSTM(128, 128, batch_first=True)
        
        # Fusion
        self.fusion = nn.Linear(256 + 128, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, image, text):
        # Image features
        img_features = self.image_cnn(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Text features
        text_embedded = self.text_embedding(text)
        text_out, _ = self.text_lstm(text_embedded)
        text_features = text_out[:, -1, :]
        
        # Concatenate
        combined = torch.cat([img_features, text_features], dim=1)
        
        # Classification
        fused = F.relu(self.fusion(combined))
        fused = self.dropout(fused)
        out = self.classifier(fused)
        
        return out


# ==================== MODEL TRAINER ====================

class DeepLearningTrainer:
    """
    Unified trainer for all deep learning models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.optimizer = None
        self.criterion = None
        self.training_history = []
    
    def setup_training(self, learning_rate: float = 0.001, 
                      loss_fn: str = 'cross_entropy'):
        """Setup optimizer and loss function"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if loss_fn == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_fn == 'mse':
            self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10) -> Dict:
        """Full training loop"""
        best_acc = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Save best model
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                self.save_checkpoint(f'best_model_epoch{epoch+1}.pth')
        
        return {
            'best_accuracy': best_acc,
            'history': self.training_history
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"Checkpoint loaded: {path}")


# ==================== FACTORY FUNCTIONS ====================

def create_medical_cnn(num_classes: int = 14, use_resnet: bool = True) -> nn.Module:
    """Create CNN for medical images"""
    if use_resnet:
        return MedicalResNet(num_classes=num_classes)
    return MedicalImageCNN(num_classes=num_classes)


def create_medical_lstm(vocab_size: int = 10000, num_classes: int = 10,
                       use_attention: bool = False, bidirectional: bool = True) -> nn.Module:
    """Create LSTM for medical text"""
    if use_attention:
        return MedicalLSTMWithAttention(vocab_size=vocab_size, num_classes=num_classes)
    elif bidirectional:
        return MedicalBiLSTM(vocab_size=vocab_size, num_classes=num_classes)
    return MedicalTextLSTM(vocab_size=vocab_size, num_classes=num_classes)


def create_vital_signs_predictor(input_dim: int = 5) -> nn.Module:
    """Create LSTM for vital signs prediction"""
    return VitalSignsLSTM(input_dim=input_dim)


def create_multimodal_model(num_classes: int = 10, vocab_size: int = 10000) -> nn.Module:
    """Create multimodal (image + text) model"""
    return MultimodalMedicalNet(num_classes=num_classes, vocab_size=vocab_size)


if __name__ == "__main__":
    # Test models
    print("="*70)
    print("DEEP LEARNING MODELS TEST")
    print("="*70)
    
    # Test CNN
    print("\n[1] Testing Medical CNN...")
    cnn = create_medical_cnn(num_classes=14, use_resnet=True)
    test_img = torch.randn(2, 3, 224, 224)
    output = cnn(test_img)
    print(f"✅ CNN output shape: {output.shape}")
    
    # Test LSTM
    print("\n[2] Testing Medical BiLSTM...")
    lstm = create_medical_lstm(vocab_size=10000, num_classes=10, bidirectional=True)
    test_text = torch.randint(0, 10000, (2, 50))
    output = lstm(test_text)
    print(f"✅ LSTM output shape: {output.shape}")
    
    # Test LSTM with Attention
    print("\n[3] Testing LSTM with Attention...")
    lstm_attn = create_medical_lstm(vocab_size=10000, num_classes=10, use_attention=True)
    output = lstm_attn(test_text)
    print(f"✅ LSTM+Attention output shape: {output.shape}")
    
    # Test Vital Signs LSTM
    print("\n[4] Testing Vital Signs LSTM...")
    vital_lstm = create_vital_signs_predictor(input_dim=5)
    test_vitals = torch.randn(2, 24, 5)  # 24 hours, 5 vital signs
    output = vital_lstm(test_vitals)
    print(f"✅ Vital Signs LSTM output shape: {output.shape}")
    
    # Test Multimodal
    print("\n[5] Testing Multimodal Model...")
    multimodal = create_multimodal_model(num_classes=10, vocab_size=10000)
    output = multimodal(test_img, test_text)
    print(f"✅ Multimodal output shape: {output.shape}")
    
    print("\n" + "="*70)
    print("✅ ALL MODELS WORKING!")
    print("="*70)
