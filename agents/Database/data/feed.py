# data/feed.py - WERSJA PANCERNA

import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Próba importu wskaźników (bezpieczna)
try:
    from data.indicators import TechnicalIndicators
    INDICATORS_AVAILABLE = True
except ImportError:
    print(" Moduł indicators niedostępny lub błąd importu ta")
    INDICATORS_AVAILABLE = False

class DataFeed:
    @staticmethod
    def get_market_data(symbol: str, tf: str = "1h", limit: int = 100):
        # 1. Inicjalizacja
        exchange = ccxt.binance({'enableRateLimit': True})
        df = pd.DataFrame()

        try:
            # --- SCENARIUSZ A: KRYPTO (Binance) ---
            if "/" in symbol:
                # Próba pobrania świec (OHLCV)
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'v'])
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                except Exception as e:
                    print(f" Błąd pobierania świec (OHLCV) dla {symbol}: {e}")
                
                # RATUNEK: Jeśli świece nie przyszły (pusty df), pobieramy chociaż AKTUALNĄ CENĘ
                # To naprawi "NO DATA" w Scannerze i HUDzie
                if df.empty:
                    print(f"🔄 Próba pobrania Tickera (Last Price) dla {symbol}...")
                    ticker = exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    # Tworzymy sztuczną ramkę danych z jedną linią, żeby system miał co czytać
                    df = pd.DataFrame([{
                        'time': datetime.now(),
                        'open': price, 'high': price, 'low': price, 'close': price, 'v': 0
                    }])

            # --- SCENARIUSZ B: AKCJE (Yahoo) ---
            else:
                df = yf.download(symbol, period="1mo", interval=tf, progress=False).reset_index()
                df.columns = [c.lower() for c in df.columns]
                df.rename(columns={"date": "time", "adj close": "close", "volume": "v"}, inplace=True)

            # --- OBRÓBKA DANYCH ---
            if df.empty:
                return pd.DataFrame()

            # Konwersja na liczby
            cols = ['open', 'high', 'low', 'close', 'v']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            # Wskaźniki (TA) - Tylko jeśli mamy wystarczająco dużo danych
            if INDICATORS_AVAILABLE and len(df) > 14:
                try:
                    df = TechnicalIndicators.add_all(df)
                except Exception:
                    pass # Ignorujemy błędy wskaźników, ważna jest cena

            return df

        except Exception as e:
            print(f" KRYTYCZNY BŁĄD FEEDA ({symbol}): {e}")
            return pd.DataFrame()
        finally:
            # Zamykamy połączenie, żeby nie wisiało
            if hasattr(exchange, 'close'):
                # W wersji sync ccxt close nie jest wymagane/awaitowane, ale dla porządku
                pass