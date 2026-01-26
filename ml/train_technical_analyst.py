"""
Train Technical Analyst Agent
Uses extracted candlestick data with indicators
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
from datetime import datetime
import logging
import json

# Configuration
DATA_DIR = "R:/Redline_Data/training_db/technical_analyst/"
MODEL_DIR = "R:/Redline_Data/ai_models/technical_analyst/"
CHECKPOINT_DIR = "R:/Redline_Data/checkpoints/"
PLAYGROUND_DIR = "R:/Redline_Data/playground/technical_analyst_001/"

# Hyperparameters
SEQ_LENGTH = 60  # Look back 60 candles
HIDDEN_SIZE = 128
NUM_LAYERS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLAYGROUND_DIR, exist_ok=True)

# Neural Network
class TechnicalAnalystLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=3):
        super(TechnicalAnalystLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)  # BUY, SELL, HOLD
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Dataset
class CandleDataset(Dataset):
    def __init__(self, data_files, seq_length=60):
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []
        
        logger.info(f"Loading {len(data_files)} data files...")
        
        for file in data_files:
            df = pd.read_csv(file)
            
            # Features: OHLCV + indicators
            features = ['open', 'high', 'low', 'close', 'volume',
                       'rsi', 'sma_20', 'sma_50', 'ema_12',
                       'bb_upper', 'bb_middle', 'bb_lower',
                       'volume_ratio']
            
            # Drop NaN rows
            df = df[features].dropna()
            
            if len(df) < seq_length + 1:
                continue
            
            # Normalize
            data = df.values
            data_normalized = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
            
            # Create sequences
            for i in range(len(data_normalized) - seq_length):
                seq = data_normalized[i:i+seq_length]
                
                # Label: future price movement
                future_price = df['close'].iloc[i+seq_length]
                current_price = df['close'].iloc[i+seq_length-1]
                price_change = (future_price - current_price) / current_price
                
                # Classify: BUY (0), HOLD (1), SELL (2)
                if price_change > 0.01:  # >1% up
                    label = 0  # BUY
                elif price_change < -0.01:  # >1% down
                    label = 2  # SELL
                else:
                    label = 1  # HOLD
                
                self.sequences.append(seq)
                self.labels.append(label)
        
        logger.info(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.LongTensor([self.labels[idx]])[0])

def train():
    logger.info("=" * 80)
    logger.info("TRAINING TECHNICAL ANALYST AGENT")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Load data
    data_files = glob.glob(os.path.join(DATA_DIR, "*_candles.csv"))
    
    if not data_files:
        logger.error("❌ No data files found!")
        return
    
    logger.info(f"Found {len(data_files)} symbol files")
    
    # Create dataset
    dataset = CandleDataset(data_files, SEQ_LENGTH)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    logger.info(f"Train: {train_size} | Val: {val_size}")
    
    # Model
    input_size = 13  # Number of features
    model = TechnicalAnalystLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
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
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Log
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] "
                   f"Train Loss: {train_loss/len(train_loader):.4f} "
                   f"Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss/len(val_loader):.4f} "
                   f"Val Acc: {val_acc:.2f}%")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(MODEL_DIR, "technical_analyst_best.pth"))
            logger.info(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 
              os.path.join(MODEL_DIR, "technical_analyst_final.pth"))
    
    # Save training history
    with open(os.path.join(PLAYGROUND_DIR, "training_history.json"), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save metadata
    metadata = {
        'agent_id': 'technical_analyst_001',
        'generation': 1,
        'trained_at': datetime.now().isoformat(),
        'best_val_acc': best_val_acc,
        'epochs': EPOCHS,
        'seq_length': SEQ_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS
    }
    
    with open(os.path.join(MODEL_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {MODEL_DIR}")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    train()
