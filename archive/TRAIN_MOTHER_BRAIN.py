"""
REDLINE MOTHER BRAIN - TRAINING LAUNCHER
========================================
Ten skrypt uruchamia trening głównego modelu AI (Mother Brain)
używając danych z dysku R: (zarówno historical jak i bulk_data).

Używa:
- PyTorch (GPU jeśli dostępne)
- Tensorboard (do logowania wyników)
"""

import os
import sys
import torch
from agents.BackendAPI.backend.ai_core import RedlineAICore

# Dodajemy 
sys.path.append(os.getcwd())

def main():
    print("🧠 REDLINE AI CORE - TRAINING MODE")
    print("==================================")
    
    # Sprawdź GPU
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU NOT DETECTED! Training will be slow (CPU mode).")
        print("   Upewnij się, że zainstalowałeś PyTorch z CUDA.")

    # Inicjalizacja AI Core w trybie treningu
    print("\n🚀 Initializing Mother Brain (Simulator Mode)...")
    
    try:
        from agents.AIBrain.ml.train_simulator import MotherBrainTrainer
        
        # Proste Menu
        print("Wybierz tryb danych:")
        print("1. Bulk Data (R:/Redline_Data/bulk_data) - wymaga DOWNLOAD_BULK_V3")
        print("2. Historical CSV (R:/Redline_Data/historical) - wymaga DOWNLOAD_DATA")
        
        # Domyślnie automatycznie wykrywa w klasie, więc po prostu uruchamiamy
        print("\nAutomatyczne wykrywanie danych...")
        
        trainer = MotherBrainTrainer(symbol="BTCUSDT", interval="1h") # Startujemy od 1h dla szybkości
        
        if trainer.load_data():
            print("Dane załadowane. Rozpoczynam trening...")
            trainer.train_loop()
        else:
            print("⚠️ Nie udało się załadować danych. Uruchom najpierw skrypt pobierania!")
            
    except ImportError as e:
        print(f"❌ Błąd importu: {e}")
    except Exception as e:
        print(f"❌ Błąd treningu: {e}")

if __name__ == "__main__":
    main()
