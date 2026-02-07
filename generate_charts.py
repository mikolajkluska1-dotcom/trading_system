import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Ścieżki
log_file = r"R:\Redline_Data\playground\tft_training_log.csv"
output_dir = r"C:\Users\Mikołaj\.gemini\antigravity\brain\3cb254e2-e9ac-47bd-a91b-9b04fbcf9079"
output_path = os.path.join(output_dir, "training_charts_v5_2.png")

# Wczytywanie danych
df = pd.read_csv(log_file)

# Tworzenie wykresu
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Wykres Accuracy
ax1.plot(df['epoch'], df['accuracy'], color='green', marker='o', linestyle='-', linewidth=2, label='Accuracy %')
ax1.set_title('Mother Brain v5.2 - Accuracy Evolution', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Wykres Loss
ax2.plot(df['epoch'], df['loss'], color='red', marker='x', linestyle='--', linewidth=2, label='Loss')
ax2.set_title('Mother Brain v5.2 - Loss Reduction', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig(output_path)
print(f"✅ Wykres zapisany w: {output_path}")
