# ml/scanner.py
import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# CORE IMPORTS
from data.feed import DataFeed
from agents.BackendAPI.backend.ai_core import RedlineAICore
from agents.AIBrain.trading.wallet import WalletManager

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s [SCANNER] %(message)s')
logger = logging.getLogger("SCANNER")

class MarketScanner:
    """
    REDLINE Market Scanner V4.4 (DEBUG MODE)
    """

    def __init__(self, ai_core: RedlineAICore):
        self.ai_core = ai_core
        # Wymuszenie czystej instancji publicznej (bez kluczy, bez błędów auth)
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        except Exception as e:
            logger.error(f"⚠️ CCXT Init Failed: {e}")
            self.exchange = None

        # Caches & State
        self.symbols_cache = []
        self.last_cache_update = 0
        self.signal_memory = {}
        self.trade_cooldowns = {}
        self.error_count = 0

    def get_dynamic_watchlist(self, limit=30):
        """Pobiera Top Płynne Aktywa z potężnym fallbackiem"""
        if time.time() - self.last_cache_update < 900 and self.symbols_cache:
            return self.symbols_cache

        try:
            if not self.exchange: raise ValueError("No Exchange Instance")
            
            # Próba pobrania Live Tickers
            tickers = self.exchange.fetch_tickers()
            valid = []

            for symbol, data in tickers.items():
                if '/USDT' not in symbol: continue
                # Filtrujemy "śmieciowe" tokeny
                if any(x in symbol for x in ['UP/', 'DOWN/', 'BEAR', 'BULL', 'TUSD', 'USDC', 'FDUSD', 'DAI', 'EUR']):
                    continue

                quote_vol = data.get('quoteVolume', 0)
                if quote_vol > 5_000_000: # Min 5M volume
                    valid.append((symbol, quote_vol))

            # Sortujemy i bierzemy Top N
            self.symbols_cache = [x[0] for x in sorted(valid, key=lambda x: x[1], reverse=True)[:limit]]
            self.last_cache_update = time.time()
            
            # Zawsze dodajemy BTC i ETH na szczyt
            if "BTC/USDT" not in self.symbols_cache: self.symbols_cache.insert(0, "BTC/USDT")
            
            logger.info(f"✅ Watchlist updated: {len(self.symbols_cache)} assets")
            return self.symbols_cache

        except Exception as e:
            logger.warning(f"⚠️ Watchlist API Failed: {e}. Using STATIC FALLBACK.")
            fallback = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
                "MATIC/USDT", "LTC/USDT", "SHIB/USDT", "TRX/USDT", "ATOM/USDT"
            ]
            self.symbols_cache = fallback
            return fallback

    def _build_context(self, df):
        """Buduje wskaźniki nawet jeśli dane są niepełne"""
        last = df.iloc[-1]
        close = float(last['close'])

        # Obliczanie wskaźników w locie (Fail-safe)
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            rsi = df['rsi'].iloc[-1]
        else:
            rsi = float(last['rsi'])

        if pd.isna(rsi): rsi = 50.0

        high = float(last['high'])
        low = float(last['low'])
        volatility = (high - low) / close if close > 0 else 0.0

        return {
            "close": close,
            "rsi": rsi,
            "volatility": volatility
        }

    def _process_single_symbol(self, symbol, config, market_quality_scalar):
        """Worker z pełnym logowaniem błędów"""
        try:
            # 1. Pobranie Danych
            df = DataFeed.get_market_data(symbol, "1h", limit=60)
            if df is None or df.empty:
                # logger.debug(f"❌ {symbol}: Empty Data") # Odkomentuj jeśli chcesz widzieć wszystko
                return None

            # 2. Context
            market_ctx = self._build_context(df)
            
            # Filter A: Zmienność (Obniżony próg dla testów)
            if market_ctx['volatility'] < 0.002: # 0.2%
                return None 

            # 3. AI Evaluation
            decision = self.ai_core.evaluate(symbol, market_ctx, df, config)
            if not decision:
                return None

            adjusted_score = decision.score * market_quality_scalar

            # 4. Final Decision Logic
            final_action = "HOLD"
            if adjusted_score >= 60: final_action = "BUY"
            elif adjusted_score <= 40: final_action = "SELL"
            
            # Tylko zwracamy ciekawe wyniki, żeby nie śmiecić
            if final_action == "HOLD" and adjusted_score > 20 and adjusted_score < 80:
                return None

            return {
                "symbol": symbol,
                "price": market_ctx["close"],
                "signal": final_action,
                "score": round(adjusted_score, 1),
                "confidence": decision.confidence,
                "reason": "; ".join(decision.reasons),
                "ts": datetime.now().strftime("%H:%M:%S")
            }

        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {str(e)}")
            return None

    def run(self, config):
        """Główna pętla"""
        start_time = time.time()
        self.error_count = 0
        
        # 1. Pobierz listę
        watchlist = self.get_dynamic_watchlist(limit=15) # Mniej par = szybciej dla testu
        
        # 2. Skanowanie równoległe
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self._process_single_symbol, symbol, config, 1.0)
                for symbol in watchlist
            ]
            for future in futures:
                res = future.result()
                if res: results.append(res)
                else: self.error_count += 1

        # 3. Logowanie Wyniku
        if not results:
            logger.warning(f"⚠️ Scan finished but returned 0 opportunities. (Errors/Skipped: {self.error_count})")
            # To może powodować wrażenie "Offline" - brak wyników
        else:
            logger.info(f"✅ Scan found {len(results)} signals.")
            
        return results
