"""
Check Training Data Class Distribution
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents.AIBrain.ml.recorder_v6 import RecorderV6

print("🔍 Analyzing Training Data Distribution...")
print("="*60)

# Create recorder (same as training)
rec = RecorderV6()

# We need to check what create_dataset produced
# Let's look at the target generation logic

loader = rec.loader
loader.index_data()

query = "SELECT * FROM klines ORDER BY filename, open_time LIMIT 10000"
df_sample = loader.con.execute(query).df()

print(f"📊 Sample size: {len(df_sample)} candles\n")

# Simulate target generation (from recorder_v6.py logic)
seq_len = 30

# Extract features
c = df_sample['close'].values
ret = np.diff(np.log(c + 1e-9), prepend=c[0])
vol = pd.Series(ret).rolling(7).std().fillna(0.01).values

# Count targets
targets = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
thresholds_used = []

for i in range(seq_len, len(df_sample)-5):
    # Calculate future return (4h ahead)
    fut_ret = (c[i+4] - c[i]) / c[i] if i+4 < len(c) else 0
    
    # Dynamic threshold (from recorder_v6.py)
    current_vol = vol[i]
    base_threshold = 0.015
    dynamic_threshold = max(base_threshold, current_vol * 2.0)
    fee_adjusted_threshold = dynamic_threshold + 0.002
    
    thresholds_used.append(fee_adjusted_threshold)
    
    # Classification
    if fut_ret > fee_adjusted_threshold:
        targets['BUY'] += 1
    elif fut_ret < -fee_adjusted_threshold:
        targets['SELL'] += 1
    else:
        targets['HOLD'] += 1

total = sum(targets.values())

print("📊 CLASS DISTRIBUTION:")
print("="*60)
for cls, count in targets.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)
    print(f"   {cls:6s}: {count:5d} ({pct:5.1f}%) {bar}")

print(f"\n📊 THRESHOLD STATISTICS:")
print(f"   Average threshold: {np.mean(thresholds_used)*100:.3f}%")
print(f"   Min threshold:     {np.min(thresholds_used)*100:.3f}%")
print(f"   Max threshold:     {np.max(thresholds_used)*100:.3f}%")
print(f"   Std threshold:     {np.std(thresholds_used)*100:.3f}%")

# Diagnosis
print("\n" + "="*60)
print("🔍 DIAGNOSIS")
print("="*60)

hold_pct = (targets['HOLD'] / total) * 100

if hold_pct > 80:
    print("❌ CRITICAL: Class imbalance!")
    print(f"   {hold_pct:.1f}% of data is HOLD")
    print("\n🔬 ROOT CAUSE:")
    print("   Dynamic thresholds (volatility + fees) are TOO HIGH")
    print("   Most price movements don't exceed the threshold")
    print("\n💡 SOLUTIONS:")
    print("   A. Lower base threshold: 1.5% → 0.5%")
    print("   B. Remove fee from threshold")
    print("   C. Use class weights in loss function")
    print("   D. Oversample BUY/SELL during training")
    
elif hold_pct > 60:
    print("⚠️  Moderate imbalance")
    print("   Can be fixed with class weights")
    
else:
    print("✅ Balanced classes!")

print("="*60)
