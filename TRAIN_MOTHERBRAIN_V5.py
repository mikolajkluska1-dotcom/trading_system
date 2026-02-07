"""
AIBrain v5.1 - Weighted Training Script (Anti-Lazy Model)
========================================================
Trening z użyciem WeightedRandomSampler oraz Weighted Cross Entropy.
Zmusza model do szukania sygnałów BUY/SELL zamiast ciągłego HOLD.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from datetime import datetime

# Fix paths for imports
# Dodajemy agents/AIBrain do ścieżki, aby ułatwić importy
sys.path.insert(0, str(Path(__file__).parent / "agents" / "AIBrain"))

# Importy z sub-modułów (teraz widoczne dzięki sys.path)
try:
    from config import MODELS_DIR, PLAYGROUND_DIR, SEQ_LEN, BATCH_SIZE, LEARNING_RATE, EPOCHS
    from ml.mother_brain_v5 import MotherBrainV5
    from ml.recorder import TrainingRecorder
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Próbuję importu z pełną ścieżką...")
    from agents.AIBrain.config import MODELS_DIR, PLAYGROUND_DIR, SEQ_LEN, BATCH_SIZE, LEARNING_RATE, EPOCHS
    from agents.AIBrain.ml.mother_brain_v5 import MotherBrainV5
    from agents.AIBrain.ml.recorder import TrainingRecorder

# Ustawienia
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = MODELS_DIR / "mother_v5_tft.pth"

# Quantile Loss (Pinball Loss)
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles # e.g., [0.1, 0.5, 0.9]

    def forward(self, predictions, targets):
        assert predictions.size(1) == len(self.quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.max(q * errors, (q - 1) * errors))
        return torch.mean(torch.sum(torch.stack(losses, dim=1), dim=1))

def main():
    print("Starting Mother Brain v5.2 Training on " + DEVICE)
    print("Loading Data and Recorder...")
    # --- 1. DATASET ---
    recorder = TrainingRecorder()
    dataset = recorder.create_dataset_from_files(seq_len=SEQ_LEN)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty. Check R:/Redline_Data/bulk_data/klines/")
        return

    # Proste ładowanie (bez balansu klas - dla regresji niepotrzebne w tym kroku)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 2. MODEL & OPTIMIZER ---
    model = MotherBrainV5() # Already initialized on DEVICE inside __init__
    
    # Próba wczytania checkpointu (jeśli istnieje)
    start_epoch = 0
    
    if os.path.exists(SAVE_PATH):
        try:
            if model.load(SAVE_PATH):
                print(f"✅ Resuming training. Best Acc so far: {model.best_accuracy:.1%}")
            else:
                print("⚠️ Checkpoint format unknown or incompatible, starting fresh.")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint (likely arch mismatch): {e}")
            print("💡 Starting fresh training with 9 agents.")

    best_acc = model.best_accuracy

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Tu wchodzi "bat" na lenistwo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # --- 3. PĘTLA TRENINGOWA ---
    print(f"Starting Training: {EPOCHS} Epochs, Batch: {BATCH_SIZE}")
    print(f"Initial Best Acc: {best_acc:.1%}")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (agents_in, context_in, lstm_in, target) in enumerate(progress):
            # Mapowanie na model (agent_signals, market_context, price_sequence)
            agents_in = agents_in.to(DEVICE)    # (batch, 9)
            context_in = context_in.to(DEVICE)  # (batch, 11)
            lstm_in = lstm_in.to(DEVICE)        # (batch, 30, 6)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            quantiles, _, _ = model(agents_in, context_in, lstm_in)
            
            loss = criterion(quantiles, target)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Metryki (Regression style)
            total_loss += loss.item()
            
            median_pred = quantiles[:, 1]
            # Directional Acc: Is the predicted sign same as actual return sign?
            sign_match = (torch.sign(median_pred) == torch.sign(target)).float()
            correct += sign_match.sum().item()
            total_samples += target.size(0)
            
            current_acc = (correct / total_samples) if total_samples > 0 else 0
            
            if batch_idx % 20 == 0:
                # Pokazujemy MAE dla mediany
                mae = torch.mean(torch.abs(median_pred - target)).item()
                progress.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'DirAcc': f"{current_acc:.1%}",
                    'MAE': f"{mae:.4f}"
                })

        # --- KONIEC EPOKI ---
        epoch_acc = correct / total_samples
        epoch_loss = total_loss / len(loader)
        
        scheduler.step(epoch_acc)
        
        # Zapisywanie
        improved = False
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            improved = True
            model.best_accuracy = best_acc # Sync best acc to wrapper
            model.save(str(SAVE_PATH))     # Use wrapper's save logic
        elif (epoch + 1) % 5 == 0:
            # Backup co 5 epok
            backup_path = str(SAVE_PATH).replace(".pth", "_backup.pth")
            model.save(backup_path)

        status_msg = "🏆 NEW BEST" if improved else "🏁 Summary"
        print(f"{status_msg}: DirAcc={epoch_acc:.1%} | Loss={epoch_loss:.4f} | Best={best_acc:.1%}")

        # --- LOGOWANIE DO DASHBOARDU ---
        log_file = PLAYGROUND_DIR / "tft_training_log.csv"
        log_data = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'loss': epoch_loss,
            'accuracy': epoch_acc * 100,
            'samples': total_samples
        }
        
        # Proste dopisywanie do CSV bez pandas dla wydajności
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', encoding='utf-8') as f:
            if not file_exists:
                f.write("epoch,timestamp,loss,accuracy,samples\n")
            f.write(f"{log_data['epoch']},{log_data['timestamp']},{log_data['loss']:.6f},{log_data['accuracy']:.2f},{log_data['samples']}\n")

if __name__ == "__main__":
    main()
