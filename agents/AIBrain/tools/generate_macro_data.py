"""
Generate Synthetic Macro Data for Mother Brain Training
======================================================
Symuluje "głos Ojca" (Father Brain) dla danych historycznych.
Ponieważ nie możemy zapytać GPT o przeszłość (brak kontekstu real-time),
generujemy syntetyczny sentyment bazując na zachowaniu ceny Bitcoina.

Zasada:
- Krach (-5%) -> Ojciec krzyczy "PANIKA" (-0.9)
- Hossa (+5%) -> Ojciec mówi "BEZPIECZNIE" (+0.8)
- Boczniak + Volatility -> Ojciec mówi "NIEPEWNOŚĆ" (-0.3)
"""
import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from agents.AIBrain.config import DATA_DIR

def calculate_synthetic_macro(df):
    """
    Oblicza syntetyczny wynik sentymentu makro (-1.0 do 1.0)
    na podstawie zachowania ceny w tym pliku (zakładamy korelację z rynkiem).
    """
    # Calculate volatility and returns
    returns = df['close'].pct_change()
    # ATR-like volatility (rolling std dev of returns)
    volatility = returns.rolling(window=24).std()
    
    macro_scores = []
    
    for i in range(len(df)):
        if i < 24:
            macro_scores.append(0.0)
            continue
            
        ret = returns.iloc[i]
        vol = volatility.iloc[i]
        
        score = 0.0
        
        # 1. Extreme Moves (Panic/Euphoria)
        if ret < -0.05:     # -5% drop
            score = -0.9    # Extreme Fear
        elif ret < -0.02:   # -2% drop
            score = -0.5    # Fear
        elif ret > 0.05:    # +5% pump
            score = 0.8     # Euphoria
        elif ret > 0.02:    # +2% pump
            score = 0.4     # Greed
            
        # 2. Uncertainty (Sideways but volatile)
        # If absolute return is small but volatility is high
        if vol > 0.02 and abs(ret) < 0.01:
            score -= 0.3    # Nervous market
            
        # 3. Stability Bonus
        # Low volatility, slow grind up
        if vol < 0.01 and ret > 0:
            score += 0.2    # Safe grind
            
        # Clamp score
        score = max(-1.0, min(1.0, float(score)))
        macro_scores.append(score)
        
    return macro_scores

def process_files():
    print("👴 Generowanie historycznych danych 'Father Brain' (Syntetycznych)...")
    
    # Przeszukaj wszystkie podkatalogi (1h, 4h, synthetic)
    search_paths = [
        str(DATA_DIR / "1h/*.csv"),
        str(DATA_DIR / "4h/*.csv"),
        str(DATA_DIR / "synthetic/*.csv")
    ]
    
    files = []
    for path in search_paths:
        files.extend(glob.glob(path))
        
    print(f"📂 Znaleziono {len(files)} plików do przetworzenia.")
    
    count = 0
    for file in tqdm(files):
        try:
            df = pd.read_csv(file)
            
            # Jeśli kolumna już istnieje, pomiń (chyba że chcemy nadpisać)
            # if 'macro_sentiment' in df.columns: continue 
            
            # Generuj dane
            df['macro_sentiment'] = calculate_synthetic_macro(df)
            
            # Zapisz
            df.to_csv(file, index=False)
            count += 1
            
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
            
    print(f"✅ Zaktualizowano {count} plików o dane makro.")

if __name__ == "__main__":
    process_files()
