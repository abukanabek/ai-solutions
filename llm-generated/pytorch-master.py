"""
3_pytorch_dl_master.py
Comprehensive PyTorch Deep Learning Pipelines
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== NEURAL NETWORK ARCHITECTURES ====================

class SimpleNN(nn.Module):
    """Simple Feedforward Neural Network"""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):
        super(SimpleNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class CNNClassifier(nn.Module):
    """CNN for image classification"""
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class LSTMModel(nn.Module):
    """LSTM for sequence data"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

class Autoencoder(nn.Module):
    """Simple Autoencoder for dimensionality reduction"""
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TransformerClassifier(nn.Module):
    """Transformer for sequence classification"""
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_seq_length):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

# ==================== TRAINING UTILITIES ====================

class PyTorchTrainer:
    """Comprehensive PyTorch training utilities"""
    
    def __init__(self, model, device='auto'):
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def _get_device(self, device):
        """Get available device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def train_classification(self, train_loader, val_loader, criterion, optimizer, 
                           scheduler=None, epochs=10, clip_grad=None):
        """Train classification model"""
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training phase
            self.model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if clip_grad:
                    clip_grad_norm_(self.model.parameters(), clip_grad)
                    
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Validation phase
            val_loss, val_acc = self.evaluate_classification(val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Store history
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}: '
                  f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    def train_regression(self, train_loader, val_loader, criterion, optimizer, 
                        scheduler=None, epochs=10):
        """Train regression model"""
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0.0
            
            # Training phase
            self.model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = self.evaluate_regression(val_loader, criterion)
            
            if scheduler:
                scheduler.step()
            
            train_loss_avg = train_loss / len(train_loader)
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f}')
    
    def evaluate_classification(self, data_loader, criterion):
        """Evaluate classification model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(data_loader)
        
        return avg_loss, accuracy
    
    def evaluate_regression(self, data_loader, criterion):
        """Evaluate regression model"""
        self.model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
        
        return test_loss / len(data_loader)
    
    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.cpu().numpy())
        
        return np.concatenate(predictions)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot (if available)
        if 'train_acc' in self.history:
            ax2.plot(self.history['train_acc'], label='Train Acc')
            ax2.plot(self.history['val_acc'], label='Val Acc')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()

# ==================== DATA LOADERS ====================

class TabularDataset(Dataset):
    """Dataset for tabular data"""
    def __init__(self, X, y=None, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        return x

class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    def __init__(self, sequences, targets=None, sequence_length=10):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences) - self.sequence_length
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx:idx + self.sequence_length]
        if self.targets is not None:
            target = self.targets[idx + self.sequence_length]
            return sequence, target
        return sequence

# ==================== COMPLETE PIPELINES ====================

class PyTorchClassificationPipeline:
    """Complete classification pipeline with PyTorch"""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64]):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = SimpleNN(input_size, hidden_sizes, num_classes)
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X, y=None):
        """Preprocess tabular data"""
        if y is not None:
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = None
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_encoded
    
    def create_data_loaders(self, X_train, X_val, y_train, y_val, batch_size=32):
        """Create PyTorch data loaders"""
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
        """Train the model"""
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train)
        X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train_scaled, X_val_scaled, y_train_encoded, y_val_encoded
        )
        
        # Initialize trainer
        self.trainer = PyTorchTrainer(self.model)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # Train model
        self.trainer.train_classification(
            train_loader, val_loader, criterion, optimizer, 
            scheduler, epochs=epochs
        )
        
        return self.trainer.history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        dataset = TabularDataset(X_scaled)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = self.trainer.predict(data_loader)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return self.label_encoder.inverse_transform(predicted_classes)

class PyTorchRegressionPipeline:
    """Complete regression pipeline with PyTorch"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64]):
        self.input_size = input_size
        self.model = SimpleNN(input_size, hidden_sizes, 1)  # Single output for regression
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def preprocess_data(self, X, y=None):
        """Preprocess data for regression"""
        X_scaled = self.scaler_x.fit_transform(X)
        
        if y is not None:
            y_reshaped = y.reshape(-1, 1)
            y_scaled = self.scaler_y.fit_transform(y_reshaped).flatten()
            return X_scaled, y_scaled
        
        return X_scaled, None
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
        """Train regression model"""
        # Preprocess data
        X_train_scaled, y_train_scaled = self.preprocess_data(X_train, y_train)
        X_val_scaled, y_val_scaled = self.preprocess_data(X_val, y_val)
        
        # Create datasets
        train_dataset = TabularDataset(X_train_scaled, y_train_scaled)
        val_dataset = TabularDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize trainer
        self.trainer = PyTorchTrainer(self.model)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Train model
        self.trainer.train_regression(
            train_loader, val_loader, criterion, optimizer, epochs=epochs
        )
        
        return self.trainer.history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler_x.transform(X)
        dataset = TabularDataset(X_scaled)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions_scaled = self.trainer.predict(data_loader)
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
        
        return predictions.flatten()

# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification, make_regression
    
    # Classification example
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    # Regression example
    X_reg, y_reg = make_regression(n_samples=1000, n_features=15, noise=0.1, random_state=42)
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Classification pipeline
    print("Training Classification Model...")
    clf_pipeline = PyTorchClassificationPipeline(input_size=20, num_classes=3)
    clf_history = clf_pipeline.train(X_train_clf, y_train_clf, X_val_clf, y_val_clf, epochs=30)
    
    # Regression pipeline
    print("\nTraining Regression Model...")
    reg_pipeline = PyTorchRegressionPipeline(input_size=15)
    reg_history = reg_pipeline.train(X_train_reg, y_train_reg, X_val_reg, y_val_reg, epochs=50)
    
    # Make predictions
    clf_preds = clf_pipeline.predict(X_val_clf)
    reg_preds = reg_pipeline.predict(X_val_reg)
    
    print(f"Classification accuracy: {np.mean(clf_preds == y_val_clf):.4f}")
    print(f"Regression MSE: {np.mean((reg_preds - y_val_reg) ** 2):.4f}")
    
    # Plot training history
    clf_pipeline.trainer.plot_training_history()
