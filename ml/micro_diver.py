import ccxt
import pandas as pd
import time
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- KONFIGURACJA SNAJPERA ---
# Celujemy w Twoją "Elitę".
# Możesz dodać więcej, ale pamiętaj: 1 sekunda to masa danych.
TARGETS = ["BNB/USDT", "SOL/USDT", "BTC/USDT", "ETH/USDT"] 

# Parametry pobierania
DAYS_BACK = 1       # Pobieramy ostatnie 24h (bezpieczne na start)
TIMEFRAME = '1s'    # Interwał 1 sekunda (High Frequency Data)

# Adres bazy danych (widziany z Windowsa/Hosta)
# Port 5435 to ten, który wystawiliśmy w docker-compose
DB_URL = "postgresql://redline_user:redline_pass@localhost:5435/redline_db"

def fetch_micro_candles(symbol):
    print(f"\n🔫 SNAJPER: Namierzanie celu {symbol} ({TIMEFRAME})...")
    
    # Inicjalizacja Binance
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Obliczamy czas startu
    since_datetime = datetime.now() - timedelta(days=DAYS_BACK)
    since = exchange.parse8601(since_datetime.isoformat())
    
    all_candles = []
    
    # Pętla pobierająca (Binance daje max 1000 świeczek na raz)
    while True:
        try:
            # Pobierz świeczki
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            
            if not candles:
                break
            
            # Dodaj do listy
            all_candles.extend(candles)
            
            # Przesuń czas startu do ostatniej pobranej świeczki + 1 sekunda
            last_time = candles[-1][0]
            since = last_time + 1000 
            
            # Przerywamy, jeśli dotarliśmy do "teraz"
            if since > exchange.milliseconds():
                break
                
            print(f"   ...magazynek: {len(all_candles)} naboi (ostatni: {pd.to_datetime(last_time, unit='ms')})", end='\r')
            
            # Krótka przerwa, żeby API nas nie zbanowało
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"\n❌ ZACIĘCIE BRONI: {e}")
            time.sleep(5) # Odczekaj chwilę przy błędzie

    print(f"\n✅ Zrzut danych dla {symbol}: {len(all_candles)} rekordów.")
    
    if len(all_candles) > 0:
        save_to_db(all_candles, symbol)

def save_to_db(candles_data, symbol):
    print(f"💾 Archiwizacja celu {symbol} w bazie danych...")
    
    # Tworzymy DataFrame
    df = pd.DataFrame(candles_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    
    # Konwersja czasu z milisekund na datę
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['symbol'] = symbol
    
    try:
        # Łączymy się z bazą
        engine = create_engine(DB_URL)
        
        # Zapisujemy (append = dopisujemy do tabeli, nie kasujemy starych)
        df.to_sql('market_candles', engine, if_exists='append', index=False)
        print(f"🏆 SUKCES: {symbol} bezpieczny w bazie TimescaleDB.")
    except Exception as e:
        print(f"💥 BŁĄD ZAPISU SQL: {e}")

if __name__ == "__main__":
    print("🚦 PROTOKÓŁ SNAJPERA URUCHOMIONY...")
    print(f"📅 Pobieranie danych z ostatnich {DAYS_BACK} dni.")
    
    for coin in TARGETS:
        fetch_micro_candles(coin)
        
    print("\n🏁 MISJA ZAKOŃCZONA. BAZA PEŁNA.")