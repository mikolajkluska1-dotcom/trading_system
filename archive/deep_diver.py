import ccxt
import pandas as pd
import time
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# --- KONFIGURACJA "DEEP DIVE" ---
DB_HOST = "127.0.0.1"
DB_PORT = "5435"
DB_NAME = "redline_db"
DB_USER = "redline_user"
DB_PASS = "redline_pass"

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ZMIANA: Pobieramy 1-minutówki (najdokładniejsze darmowe dane)
TIMEFRAME = '1m'
# ZMIANA: Pobieramy 1000 dni wstecz (prawie 3 lata)
# 1000 dni * 1440 minut = 1.44 miliona świeczek na JEDNEGO coina!
DAYS_BACK = 1000 

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 
    'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'MATIC/USDT', 'LTC/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT',
    'PEPE/USDT', 'SHIB/USDT', 'FET/USDT', 'RNDR/USDT', 'ARB/USDT'
]

def get_engine():
    return create_engine(DB_URL)

def deep_dive():
    print(f"🌊 URUCHAMIAM DEEP DIVE (NOCNA ZMIANA)")
    print(f"🎯 Cel: {len(SYMBOLS)} par | Interwał: {TIMEFRAME} | Historia: {DAYS_BACK} dni")
    print(f"💾 Zapis do: Dysk R: (TimescaleDB)")
    print("-" * 50)

    exchange = ccxt.binance({'enableRateLimit': True})
    engine = get_engine()

    # Data startowa (np. 3 lata temu)
    start_date = datetime.now() - timedelta(days=DAYS_BACK)
    start_ts = int(start_date.timestamp() * 1000)

    total_downloaded = 0

    for symbol in SYMBOLS:
        print(f"\n⚓ Rozpoczynam pobieranie: {symbol}")
        
        # WYMUSZAMY start od daty początkowej (ignorujemy to co jest w bazie)
        current_since = start_ts

        coin_total = 0
        
        while True:
            try:
                # Binance pozwala na max 1000 świeczek w jednym zapytaniu
                ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=current_since, limit=1000)
                
                if not ohlcv:
                    print(f"   ✅ Koniec danych dla {symbol}.")
                    break

                # Konwersja
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df['symbol'] = symbol

                # Zapis do SQL (z ignorowaniem błędów duplikatów)
                # Używamy pętli try-except przy zapisie, bo to najbezpieczniejsze przy nocnym mieleniu
                try:
                    df.to_sql('market_candles', engine, if_exists='append', index=False, method='multi', chunksize=1000)
                    
                    count = len(df)
                    coin_total += count
                    total_downloaded += count
                    
                    # Logowanie postępu co paczkę
                    last_date = df['time'].iloc[-1]
                    print(f"   📥 Pobrano: {count} | Razem: {coin_total} | Data: {last_date}")

                    # Przesuwamy czas
                    last_ts_batch = ohlcv[-1][0]
                    
                    # Jeśli dotarliśmy do "teraz" (mniej niż 1 minutę temu)
                    if (datetime.now().timestamp() * 1000) - last_ts_batch < 60000:
                        print("   ⏱️ Dotarliśmy do czasu teraźniejszego.")
                        break
                        
                    current_since = last_ts_batch + 1 # +1ms żeby nie dublować

                except Exception as e:
                    # Czasami baza rzuci błędem "Unique Constraint", jeśli coś się nałoży
                    # Ignorujemy i lecimy dalej, przesuwając czas
                    print(f"   ⚠️ Konflikt danych (pomijam paczkę): {e}")
                    # Awaryjne przesunięcie czasu, żeby nie wpaść w pętlę nieskończoną
                    current_since += (1000 * 60 * 1000) # Skocz o 1000 minut do przodu

                # Krótka pauza dla API (nie chcemy bana w nocy)
                time.sleep(0.2)

            except Exception as e:
                print(f"   ❌ Błąd sieci/API: {e}")
                print("   💤 Czekam 10 sekund...")
                time.sleep(10)

    print("=" * 50)
    print(f"🎉 WIELKIE POBIERANIE ZAKOŃCZONE.")
    print(f"📦 Łącznie dodano: {total_downloaded} wierszy.")
    print("Dysk R: jest teraz pełen wiedzy.")

if __name__ == "__main__":
    deep_dive()