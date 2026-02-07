"""
Synthetic Father Data Injector (v5.0)
=====================================
Wstrzykuje "symulowany sentyment Ojca" (father_score) do danych historycznych.
Kluczowe dla treningu Mother Brain v5, aby nauczyła się słuchać Ojca.

Logika:
Tata jest Mądry Po Szkodzie (Hindsight Wisdom):
- Cena BTC mocno spada? Tata "wiedział" i dawał -1.0.
- Cena BTC rośnie? Tata "wiedział" i dawał +0.8.
"""
import pandas as pd
import glob
import os
from tqdm import tqdm
import sys

# Dodajemy ścieżkę do projektu, aby zaimportować config
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from agents.AIBrain.config import DATA_DIR

def generate_synthetic_sentiment(df):
    """
    Generuje synthetic father_score (-1.0 do 1.0)
    bazując na przyszłej wiedzy (hindsight) lub wskaźnikach trendu.
    """
    # Obliczamy lokalny trend (np. SMA 50)
    df['sma50'] = df['close'].rolling(50).mean()
    df['returns'] = df['close'].pct_change()
    
    father_scores = []
    
    for i in range(len(df)):
        if i < 50:
            father_scores.append(0.0)
            continue
            
        close = df.iloc[i]['close']
        sma = df.iloc[i]['sma50']
        ret = df.iloc[i]['returns']
        
        score = 0.0
        
        # Logika "Mądrości po szkodzie" (Idealna do treningu korelacji)
        if close > sma * 1.05:          # +5% nad średnią
            score = 0.8                 # Strong Bull trend
        elif close > sma:
            score = 0.3                 # Mild Bull
        elif close < sma * 0.95:        # -5% pod średnią
            score = -0.8                # Strong Bear trend
        elif close < sma:
            score = -0.3                # Mild Bear
        
        # Panic detection (Crash protection)
        if ret < -0.03: 
            score = -1.0                # Krach! - Tata krzyczy "SPRZEDAWAJ"
        
        father_scores.append(score)
        
    return father_scores

def run_injection():
    print(f"👴 Father Injector: Scanning files in {DATA_DIR}...")
    
    # Obsługujemy różne podfoldery z configu
    search_paths = [
        str(DATA_DIR / "1h/*.csv"),
        str(DATA_DIR / "4h/*.csv"),
        str(DATA_DIR / "synthetic/*.csv")
    ]
    
    files = []
    for path in search_paths:
        found = list(glob.glob(path))
        files.extend(found)
        
    print(f"💉 Injecting Synthetic Father Data into {len(files)} files...")
    
    count = 0
    for file in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(file)
            
            # Generuj/Nadpisz father_score
            df['father_score'] = generate_synthetic_sentiment(df)
            
            # Usuń tymczasowe kolumny jeśli nie są potrzebne
            if 'sma50' in df.columns:
                df = df.drop(columns=['sma50'])
            
            # Zapisz z powrotem
            df.to_csv(file, index=False)
            count += 1
        except Exception as e:
            print(f"❌ Error in {file}: {e}")
            
    print(f"✅ Success! Updated {count} files.")

if __name__ == "__main__":
    run_injection()
