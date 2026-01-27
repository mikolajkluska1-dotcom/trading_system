"""
VOLUME HUNTER TRAINING - Day 2
Focuses on volume anomalies and patterns
Uses CNN architecture for pattern recognition
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from turbo_loader import TurboDataLoader

logger = logging.getLogger("VOLUME_HUNTER")
logging.basicConfig(level=logging.INFO)

# CONFIGURATION
BATCH_SIZE = 2048
EPOCHS = 1000
SEQ_LENGTH = 60
LEARNING_RATE = 0.0003
CONV_CHANNELS = [32, 64, 128]  # CNN channels
FC_SIZE = 256
DROPOUT = 0.3

# Paths
MODEL_DIR = "R:/Redline_Data/ai_models/volume_hunter/"
CHECKPOINT_DIR = "R:/Redline_Data/checkpoints/volume_hunter/"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class VolumeCNN(nn.Module):
    """
    CNN-based model for volume pattern recognition
    Better than LSTM for spatial patterns in volume data
    """
    
    def __init__(self, input_channels, seq_length, output_size=3):
        super(VolumeCNN, self).__init__()
        
        # 1D Convolutional layers for time series
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(DROPOUT)
        
        # Calculate flattened size
        conv_output_size = 128 * (seq_length // 8)  # After 3 pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, FC_SIZE)
        self.fc2 = nn.Linear(FC_SIZE, 128)
        self.fc3 = nn.Linear(128, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class VolumeDataset(Dataset):
    """
    Dataset focused on volume patterns
    """
    
    def __init__(self, data_loader, symbols, seq_length=60, samples_per_symbol=10000):
        self.loader = data_loader
        self.symbols = symbols
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []
        
        logger.info(f"🎲 Creating Volume Hunter dataset...")
        
        # Volume-focused features
        self.features = [
            'volume',           # Raw volume
            'volume_ratio',     # Volume vs average
            'volume_sma',       # Volume moving average
            'close',            # Price (for context)
            'high', 'low',      # Price range
            'rsi',              # Momentum
            'macd'              # Trend
        ]
        
        total_sequences = 0
        
        for symbol in symbols:
            logger.info(f"   Processing {symbol}...")
            
            windows = self.loader.get_random_window(
                symbol, 
                window_size=seq_length + 1,
                n_samples=samples_per_symbol
            )
            
            if not windows:
                continue
            
            for window in windows:
                if len(window) < seq_length + 1:
                    continue
                
                try:
                    feature_data = window[self.features].values
                    
                    # Normalize
                    feature_data = (feature_data - feature_data.mean(axis=0)) / (feature_data.std(axis=0) + 1e-8)
                    
                    seq = feature_data[:seq_length]
                    
                    # Label based on VOLUME SPIKE + PRICE MOVEMENT
                    current_vol = window['volume'].iloc[seq_length - 1]
                    avg_vol = window['volume_sma'].iloc[seq_length - 1]
                    vol_spike = current_vol / avg_vol if avg_vol > 0 else 1
                    
                    current_price = window['close'].iloc[seq_length - 1]
                    future_price = window['close'].iloc[seq_length]
                    price_change = (future_price - current_price) / current_price
                    
                    # Volume spike + price up = BUY signal
                    # Volume spike + price down = SELL signal
                    if vol_spike > 1.5 and price_change > 0.01:
                        label = 0  # BUY (volume confirms uptrend)
                    elif vol_spike > 1.5 and price_change < -0.01:
                        label = 2  # SELL (volume confirms downtrend)
                    else:
                        label = 1  # HOLD (no clear volume signal)
                    
                    self.sequences.append(seq)
                    self.labels.append(label)
                    total_sequences += 1
                    
                except Exception as e:
                    continue
        
        logger.info(f"✅ Volume dataset created: {total_sequences:,} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )

def train_volume_hunter():
    """
    Train Volume Hunter with CNN architecture
    """
    logger.info("=" * 80)
    logger.info("📊 VOLUME HUNTER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Architecture: CNN (Conv1D)")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # Load data
    logger.info("\n📊 Loading data...")
    loader = TurboDataLoader()
    data = loader.load_all_data()
    data = loader.calculate_indicators()
    
    # Create dataset
    logger.info("\n🎲 Creating dataset...")
    symbols = loader.get_all_symbols()
    dataset = VolumeDataset(loader, symbols, seq_length=SEQ_LENGTH, samples_per_symbol=10000)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"\n📊 Dataset: Train={train_size:,}, Val={val_size:,}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    # Model
    input_channels = len(dataset.features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️  Device: {device}")
    
    model = VolumeCNN(input_channels, SEQ_LENGTH).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model Parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_acc = 0.0
    training_history = []
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING VOLUME HUNTER TRAINING")
    logger.info("=" * 80)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%"
            )
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, BEST_MODEL_PATH)
            logger.info(f"  🏆 NEW BEST! Val Acc: {val_acc:.2f}%")
        
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, checkpoint_path)
    
    # Save history
    with open(os.path.join(MODEL_DIR, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ VOLUME HUNTER TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"🏆 Best Accuracy: {best_val_acc:.2f}%")
    logger.info(f"⏱️  Time: {elapsed / 3600:.2f}h")
    logger.info("=" * 80)

if __name__ == "__main__":
    train_volume_hunter()
