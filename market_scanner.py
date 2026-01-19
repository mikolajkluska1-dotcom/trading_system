import ccxt
import pandas as pd
import time
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- KONFIGURACJA ---
# Używamy portu 5435, który zadziałał!
DB_HOST = "127.0.0.1"
DB_PORT = "5435" 
DB_NAME = "redline_db"
DB_USER = "redline_user"
DB_PASS = "redline_pass"

# URL do połączenia dla SQLAlchemy
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Parametry pobierania
DAYS_BACK = 365   # Pobieramy rok wstecz
TIMEFRAME = '1h'  # Świeczki 1-godzinne
# Lista coinów do pobrania (Możesz dodać więcej)
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT'
]

def fetch_and_save_history():
    print(f"🚀 Uruchamiam REDLINE BACKFILLER")
    print(f"📡 Łączę się z bazą na porcie {DB_PORT}...")
    
    try:
        engine = create_engine(DB_URL)
        # Test połączenia
        with engine.connect() as conn:
            print("✅ Połączenie z bazą aktywne!")
    except Exception as e:
        print(f"❌ Błąd połączenia z bazą: {e}")
        return

    print(f"🌍 Giełda: Binance | Pary: {len(SYMBOLS)} | Dni: {DAYS_BACK}")
    print("-" * 50)

    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Obliczamy start (rok temu) w milisekundach
    since_date = datetime.now() - timedelta(days=DAYS_BACK)
    start_timestamp = int(since_date.timestamp() * 1000)

    total_records = 0

    for symbol in SYMBOLS:
        print(f"\n🔍 Pobieranie: {symbol}...")
        current_since = start_timestamp
        symbol_records = 0
        
        while True:
            try:
                # Pobierz 1000 świeczek
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current_since, limit=1000)
                
                if not ohlcv:
                    break 

                # Tworzenie tabeli danych (DataFrame)
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                
                # Konwersja czasu (z liczb na daty)
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df['symbol'] = symbol

                # Zapis do bazy danych
                df.to_sql('market_candles', engine, if_exists='append', index=False, method='multi', chunksize=1000)
                
                count = len(df)
                symbol_records += count
                total_records += count
                
                # Przesunięcie czasu do przodu
                last_ts = ohlcv[-1][0]
                current_since = last_ts + 1
                
                print(f"   💾 Zapisano {count} świeczek (Data: {df['time'].iloc[-1]})")

                # Jeśli pobrał mniej niż 1000, to znaczy że doszedł do dzisiaj
                if len(ohlcv) < 1000:
                    break
                
                # Krótka pauza dla API
                time.sleep(0.1)

            except Exception as e:
                print(f"⚠️ Błąd: {e}")
                time.sleep(2)

        print(f"🏁 {symbol} zakończony. Pobrano: {symbol_records} świeczek.")

    print("=" * 50)
    print(f"🎉 ZAKOŃCZONO! Łącznie w bazie: {total_records} nowych rekordów.")

if __name__ == "__main__":
    fetch_and_save_history()