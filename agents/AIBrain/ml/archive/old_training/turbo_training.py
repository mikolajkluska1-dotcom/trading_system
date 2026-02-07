"""
TURBO TRAINING - Maximum Performance Mode
1000 Epochs | Batch Size 2048 | Full RAM Utilization
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
from turbo_loader import TurboDataLoader

logger = logging.getLogger("TURBO_TRAINING")
logging.basicConfig(level=logging.INFO)

# TURBO CONFIGURATION
BATCH_SIZE = 2048  # 🔥 32x larger than default!
EPOCHS = 1000      # 🔥 20x more than default!
SEQ_LENGTH = 60
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 256  # Bigger model for more data
NUM_LAYERS = 4     # Deeper network
DROPOUT = 0.3

# Paths
MODEL_DIR = "R:/Redline_Data/ai_models/turbo_technical_analyst/"
CHECKPOINT_DIR = "R:/Redline_Data/checkpoints/turbo/"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Enhanced LSTM for bigger dataset
class TurboLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout=0.3):
        super(TurboLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # Can enable for even more power
        )
        
        # Deeper FC layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class TurboDataset(Dataset):
    """
    Ultra-fast dataset using pre-loaded data from RAM
    Uses experience replay for variety
    """
    
    def __init__(self, data_loader, symbols, seq_length=60, samples_per_symbol=10000):
        self.loader = data_loader
        self.symbols = symbols
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []
        
        logger.info(f"🎲 Creating dataset with Experience Replay...")
        logger.info(f"   Symbols: {len(symbols)}")
        logger.info(f"   Samples per symbol: {samples_per_symbol}")
        
        # Features to use
        self.features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower',
            'volume_ratio'
        ]
        
        total_sequences = 0
        
        for symbol in symbols:
            logger.info(f"   Processing {symbol}...")
            
            # Get random windows (Experience Replay!)
            windows = self.loader.get_random_window(
                symbol, 
                window_size=seq_length + 1,  # +1 for label
                n_samples=samples_per_symbol
            )
            
            if not windows:
                logger.warning(f"   Skipping {symbol} - insufficient data")
                continue
            
            for window in windows:
                if len(window) < seq_length + 1:
                    continue
                
                # Extract features
                try:
                    feature_data = window[self.features].values
                    
                    # Normalize
                    feature_data = (feature_data - feature_data.mean(axis=0)) / (feature_data.std(axis=0) + 1e-8)
                    
                    # Sequence (first seq_length rows)
                    seq = feature_data[:seq_length]
                    
                    # Label (price movement prediction)
                    current_price = window['close'].iloc[seq_length - 1]
                    future_price = window['close'].iloc[seq_length]
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
                    total_sequences += 1
                    
                except Exception as e:
                    continue
        
        logger.info(f"✅ Dataset created: {total_sequences:,} sequences")
        logger.info(f"   Features: {len(self.features)}")
        logger.info(f"   Sequence length: {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )

def train_turbo():
    """
    TURBO TRAINING - 1000 Epochs with Full RAM Power
    """
    logger.info("=" * 80)
    logger.info("🔥 TURBO TRAINING MODE ACTIVATED")
    logger.info("=" * 80)
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Hidden Size: {HIDDEN_SIZE}")
    logger.info(f"Layers: {NUM_LAYERS}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # 1. Load ALL data into RAM
    logger.info("\n📊 STEP 1: Loading entire dataset into RAM...")
    loader = TurboDataLoader()
    data = loader.load_all_data()
    data = loader.calculate_indicators()
    
    # Memory stats
    stats = loader.get_memory_stats()
    logger.info(f"💾 Dataset in RAM: {stats['dataset_gb']:.2f} GB")
    logger.info(f"💾 RAM Available: {stats['available_ram_gb']:.2f} GB")
    
    # 2. Create dataset with Experience Replay
    logger.info("\n🎲 STEP 2: Creating training dataset with Experience Replay...")
    symbols = loader.get_all_symbols()
    logger.info(f"Found {len(symbols)} symbols: {symbols}")
    
    dataset = TurboDataset(
        loader, 
        symbols, 
        seq_length=SEQ_LENGTH,
        samples_per_symbol=10000  # 10k samples per symbol!
    )
    
    # 3. Split train/val
    train_size = int(0.9 * len(dataset))  # 90% train, 10% val
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"\n📊 Dataset Split:")
    logger.info(f"   Train: {train_size:,}")
    logger.info(f"   Val: {val_size:,}")
    
    # 4. Create DataLoaders with HUGE batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Parallel loading
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"   Batches per epoch: {len(train_loader)}")
    
    # 5. Create model
    input_size = len(dataset.features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️  Device: {device}")
    
    model = TurboLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model Parameters: {total_params:,}")
    
    # 6. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 7. Training loop
    best_val_acc = 0.0
    training_history = []
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING TURBO TRAINING - 1000 EPOCHS")
    logger.info("=" * 80)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
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
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%"
            )
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save BEST model only
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, BEST_MODEL_PATH)
            
            logger.info(f"  🏆 NEW BEST MODEL! Val Acc: {val_acc:.2f}%")
        
        # Checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, checkpoint_path)
            logger.info(f"  💾 Checkpoint saved: epoch {epoch+1}")
    
    # 8. Save training history
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 9. Save metadata
    elapsed = (datetime.now() - start_time).total_seconds()
    
    metadata = {
        'model': 'TurboLSTM',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'best_val_acc': best_val_acc,
        'total_params': total_params,
        'training_time_hours': elapsed / 3600,
        'dataset_size_gb': stats['dataset_gb'],
        'symbols': symbols,
        'features': dataset.features,
        'completed_at': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TURBO TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"⏱️  Total Time: {elapsed / 3600:.2f} hours")
    logger.info(f"💾 Model saved to: {BEST_MODEL_PATH}")
    logger.info("=" * 80)

if __name__ == "__main__":
    train_turbo()
