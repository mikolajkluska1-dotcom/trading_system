"""
🚀 AI TRAINING LAUNCHER
Uruchom ten skrypt aby wystartować trening AI na całą noc.
"""

import subprocess
import sys
from datetime import datetime

print("=" * 60)
print("🧠 REDLINE AI TRAINING LAUNCHER")
print("=" * 60)
print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Wybór modelu do treningu
print("Wybierz model do treningu:")
print("1. Technical Analyst (LSTM - analiza świec)")
print("2. Volume Hunter (CNN - analiza wolumenu)")
print("3. OBA (równolegle - wymaga 64GB RAM!)")
print()

choice = input("Wybór (1/2/3): ").strip()

if choice == "1":
    print("\n🎯 Uruchamiam Technical Analyst...")
    print("📊 Dataset: 13M candles")
    print("⚡ Batch Size: 2048")
    print("🔄 Epochs: 1000")
    print("⏱️ Szacowany czas: 10-14 godzin")
    print("\n▶️ Startuje trening...\n")
    subprocess.run([sys.executable, "START_TURBO_TRAINING_AT_22.py"])

elif choice == "2":
    print("\n🎯 Uruchamiam Volume Hunter...")
    print("📊 Dataset: 13M candles")
    print("⚡ Batch Size: 2048")
    print("🔄 Epochs: 1000")
    print("⏱️ Szacowany czas: 10-14 godzin")
    print("\n▶️ Startuje trening...\n")
    subprocess.run([sys.executable, "START_VOLUME_HUNTER_AT_22.py"])

elif choice == "3":
    print("\n🎯 Uruchamiam OBA MODELE równolegle...")
    print("⚠️ UWAGA: To wymaga 64GB RAM!")
    print("📊 Dataset: 13M candles x2")
    print("⏱️ Szacowany czas: 10-14 godzin")
    
    confirm = input("\nKontynuować? (tak/nie): ").strip().lower()
    if confirm in ['tak', 't', 'yes', 'y']:
        print("\n▶️ Startuje oba treningi...\n")
        # Uruchom oba w osobnych procesach
        import threading
        
        def run_technical():
            subprocess.run([sys.executable, "START_TURBO_TRAINING_AT_22.py"])
        
        def run_volume():
            subprocess.run([sys.executable, "START_VOLUME_HUNTER_AT_22.py"])
        
        t1 = threading.Thread(target=run_technical)
        t2 = threading.Thread(target=run_volume)
        
        t1.start()
        t2.start()
        
        print("✅ Oba treningi wystartowały!")
        print("📝 Sprawdź logi w osobnych oknach")
        
        t1.join()
        t2.join()
    else:
        print("❌ Anulowano")
else:
    print("❌ Nieprawidłowy wybór")

print("\n" + "=" * 60)
print("✅ LAUNCHER ZAKOŃCZONY")
print("=" * 60)
