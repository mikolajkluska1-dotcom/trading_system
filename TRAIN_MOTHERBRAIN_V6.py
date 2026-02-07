import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Fix importów - plik jest w root projektu
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Dodajemy agents/AIBrain do path dla modułów wewnętrznych
sys.path.append(str(PROJECT_ROOT / "agents" / "AIBrain"))

try:
    from agents.AIBrain.config import MODELS_DIR, BATCH_SIZE, EPOCHS
    from agents.AIBrain.ml.mother_brain_v6 import MotherBrainV6
    from agents.AIBrain.ml.recorder_v6 import RecorderV6
except ImportError:
    # Fallback jeśli ścieżki są inne
    from config import MODELS_DIR, BATCH_SIZE, EPOCHS
    from ml.mother_brain_v6 import MotherBrainV6
    from ml.recorder_v6 import RecorderV6

def quantile_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
    # preds: [Batch, 3], target: [Batch]
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
    return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

def train_v6():
    print("🚀 STARTING V6 SOTA TRAINING (TFT + Quantiles)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Training on: {device}")
    
    # Data preparation
    rec = RecorderV6()
    ds = rec.create_dataset()
    if not ds:
        print("❌ Dataset preparation failed!")
        return
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📊 Dataset: {len(train_ds)} train, {len(val_ds)} validation")
    
    # Model setup
    model = MotherBrainV6().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # CLASS WEIGHTS - Combat HOLD bias (MODERATE approach)
    # Dataset: ~81% HOLD, 9% BUY, 9% SELL
    # Strategy: Gentle penalty for HOLD, moderate reward for trading
    class_weights = torch.tensor([0.7, 2.0, 2.0]).to(device)  
    crit_cls = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"⚖️  Class weights: HOLD={class_weights[0]:.1f}, BUY={class_weights[1]:.1f}, SELL={class_weights[2]:.1f}")
    print(f"   Ratio: {class_weights[1].item()/class_weights[0].item():.1f}x")
    
    # Tracking
    best_val_acc = 0.0
    log_file = Path("R:/Redline_Data/playground/v6_training_log.csv")
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write("epoch,timestamp,train_loss,train_acc,val_loss,val_acc,lr\n")
    
    print("🎯 Starting training loop...\n")
    
    for epoch in range(EPOCHS):
        # ============= TRAINING =============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        for price, agents, ctx, t_cls, t_reg in loop:
            price, agents, ctx = price.to(device), agents.to(device), ctx.to(device)
            t_cls, t_reg = t_cls.to(device), t_reg.to(device)
            
            optimizer.zero_grad()
            out = model(price, agents, ctx)
            
            # Multi-task loss
            l_cls = crit_cls(out['scalp'], t_cls)
            l_reg = quantile_loss(out['quantiles'], t_reg)
            loss = l_cls + 0.5 * l_reg
            
            loss.backward()
            
            # Gradient clipping (Fix #3)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(out['scalp'], 1)
            train_total += t_cls.size(0)
            train_correct += (predicted == t_cls).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*train_correct/train_total)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # ============= VALIDATION =============
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for price, agents, ctx, t_cls, t_reg in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]"):
                price, agents, ctx = price.to(device), agents.to(device), ctx.to(device)
                t_cls, t_reg = t_cls.to(device), t_reg.to(device)
                
                out = model(price, agents, ctx)
                
                l_cls = crit_cls(out['scalp'], t_cls)
                l_reg = quantile_loss(out['quantiles'], t_reg)
                loss = l_cls + 0.5 * l_reg
                
                val_loss += loss.item()
                _, predicted = torch.max(out['scalp'], 1)
                val_total += t_cls.size(0)
                val_correct += (predicted == t_cls).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging to CSV
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{timestamp},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f},{current_lr:.6f}\n")
        
        # Console output
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS}:")
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"   LR: {current_lr:.6f}\n")
        
        # Save best model (Fix #2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODELS_DIR, exist_ok=True)
            model.save(MODELS_DIR / "mother_v6_sota.pth")
            print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%\n")
    
    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📁 Model saved to: {MODELS_DIR / 'mother_v6_sota.pth'}")
    print(f"📊 Training log: {log_file}")

if __name__ == "__main__":
    train_v6()
