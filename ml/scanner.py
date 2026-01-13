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
from backend.ai_core import RedlineAICore
from trading.wallet import WalletManager

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s [SCANNER] %(message)s')
logger = logging.getLogger("SCANNER")

class MarketScanner:
    """
    REDLINE Market Scanner V4.3 (n8n & Config Ready)

    Fixes:
    - Force Public API for Tickers (Fixes '5 crypto' issue)
    - Expanded Fallback List (20+ assets)
    - Robust Data Fetching (Fixes 'Neutral' issue)
    - Scalar Logic + Multi-threading preserved
    - Context Building for BTC Analysis fixed
    """

    def __init__(self, ai_core: RedlineAICore):
        self.ai_core = ai_core
        # Wymuszenie czystej instancji publicznej (bez kluczy, bez błędów auth)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Caches & State
        self.symbols_cache = []
        self.last_cache_update = 0
        self.btc_state = {"score": 50, "trend": "NEUTRAL", "volatility": 0.0}

        # Pamięć operacyjna
        self.signal_memory = {}
        self.trade_cooldowns = {}
        self.error_count = 0

    def get_dynamic_watchlist(self, limit=30):
        """Pobiera Top Płynne Aktywa z potężnym fallbackiem"""
        # Cache 15 min
        if time.time() - self.last_cache_update < 900 and self.symbols_cache:
            return self.symbols_cache

        try:
            # Próba pobrania Live Tickers
            tickers = self.exchange.fetch_tickers()
            valid = []

            for symbol, data in tickers.items():
                if '/USDT' not in symbol: continue
                # Filtrujemy "śmieciowe" suffixy
                if any(x in symbol for x in ['UP/', 'DOWN/', 'BEAR', 'BULL', 'TUSD', 'USDC', 'FDUSD', 'DAI', 'EUR', 'GBP', 'AUD']):
                    continue

                quote_vol = data.get('quoteVolume', 0)
                if quote_vol > 5_000_000: # Min 5M volume
                    valid.append((symbol, quote_vol))

            # Sortujemy i bierzemy Top N
            self.symbols_cache = [x[0] for x in sorted(valid, key=lambda x: x[1], reverse=True)[:limit]]
            self.last_cache_update = time.time()

            if not self.symbols_cache: raise ValueError("Empty list from API")

            logger.info(f"✅ Watchlist updated: {len(self.symbols_cache)} assets")
            return self.symbols_cache

        except Exception as e:
            logger.error(f"⚠️ Watchlist API Failed: {e}. Using Extended Fallback.")
            # Rozszerzona lista awaryjna (20 par), żeby skaner nie był pusty
            fallback = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
                "MATIC/USDT", "LTC/USDT", "SHIB/USDT", "TRX/USDT", "ATOM/USDT",
                "UNI/USDT", "ETC/USDT", "FIL/USDT", "NEAR/USDT", "APT/USDT"
            ]
            self.symbols_cache = fallback
            return fallback

    def _check_position_exposure(self, symbol):
        """Sprawdza czy mamy asset w portfelu"""
        try:
            assets = WalletManager.get_assets()
            for pos in assets:
                if pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0:
                    return True
            return False
        except:
            return True # Safety

    def _calculate_position_size(self, score, quality):
        """Scalar Logic"""
        BASE_SIZE = 100.0 # To można wyciągnąć do configu

        if score >= 85: return BASE_SIZE * 1.5, "SNIPER (150%)"
        elif score >= 65: return BASE_SIZE * 1.0, "NORMAL (100%)"
        elif score >= 45: return BASE_SIZE * 0.4, "PROBE (40%)"
        else: return 0.0, "HOLD"

    def _execute_trade(self, symbol, side, size, trade_type, score, reason, decision_id, config):
        """Safe Execution Contract"""
        if not config.get("execution_enabled", False):
            logger.warning(f" DRY-RUN: {trade_type} {side} {symbol} (${size:.2f}) | Score: {score:.1f}")
            return False

        logger.info(f"🚀 LIVE EXECUTION: {side} {symbol} | Size: ${size:.2f} ({trade_type}) | ID: {decision_id}")
        # Tu kod API do składania zleceń
        self.trade_cooldowns[symbol] = time.time() + 1800
        return True

    def _build_context(self, df):
        """Buduje wskaźniki nawet jeśli dane są niepełne"""
        last = df.iloc[-1]
        close = float(last['close'])

        # Obliczanie wskaźników w locie (Fail-safe)
        if 'rsi' not in df.columns:
            # Bardzo proste proxy RSI jeśli DataFeed nie zwrócił
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            rsi = df['rsi'].iloc[-1]
        else:
            rsi = float(last['rsi'])

        # Fix dla NaN RSI (np. za mało świec)
        if pd.isna(rsi): rsi = 50.0

        high = float(last['high'])
        low = float(last['low'])
        volatility = (high - low) / close if close > 0 else 0.0

        # SMA calculation
        sma50 = float(last['sma_50']) if 'sma_50' in df.columns else close

        return {
            "close": close,
            "rsi": rsi,
            "sma_50": sma50,
            "volatility": volatility
        }

    def _process_single_symbol(self, symbol, config, market_quality_scalar):
        """Worker dla wątku"""
        try:
            if symbol in self.trade_cooldowns:
                if time.time() < self.trade_cooldowns[symbol]: return None
                del self.trade_cooldowns[symbol]

            # 2. Pobranie Danych (MTF: 1H + 4H)
            df = DataFeed.get_market_data(symbol, "1h", limit=60)
            if df.empty: return None

            # MTF Feature Injection
            try:
                df_4h = DataFeed.get_market_data(symbol, "4h", limit=50)
                if not df_4h.empty:
                    # Obliczamy trend na 4h
                    sma_50_4h = df_4h['close'].rolling(50).mean().iloc[-1]
                    last_close_4h = df_4h['close'].iloc[-1]
                    
                    # 1 = Bullish, -1 = Bearish
                    htf_trend_val = 1.0 if last_close_4h > sma_50_4h else -1.0
                    
                    # Wstrzykujemy jako stały bias do danych 1H (dla sieci neuronowej)
                    df['htf_bias'] = htf_trend_val
                else:
                    df['htf_bias'] = 0.0 # Neutral
            except:
                df['htf_bias'] = 0.0

            # 3. Hybrid Pre-Filter (Lightweight)
            # Zanim odpalimy ciężkie AI, sprawdźmy czy w ogóle warto
            market_ctx = self._build_context(df)
            
            # Filter A: Martwy Wolumen (brak zmienności)
            if market_ctx['volatility'] < 0.005: # < 0.5% ruchu 
                return None # Skip useless computation

            # 4. Decyzja AI + Scalar Logic
            decision = self.ai_core.evaluate(symbol, market_ctx, df, config)

            adjusted_score = decision.score * market_quality_scalar

            # Penalties
            penalties = []
            if market_ctx['volatility'] > 0.05:
                adjusted_score *= 0.8
                penalties.append("Vol")

            if self._check_position_exposure(symbol):
                adjusted_score *= 0.5
                penalties.append("Exp")

            adjusted_score = min(100.0, adjusted_score)

            # 4. Sizing
            trade_size, trade_type = self._calculate_position_size(adjusted_score, market_quality_scalar)

            final_action = "HOLD"
            if trade_size > 0:
                final_action = "BUY" if "BUY" in decision.action else "SELL"
                if trade_type == "SNIPER": final_action = "STRONG_" + final_action

            # 5. Auto Exec
            is_fresh = True
            cached = self.signal_memory.get(symbol)
            if cached and cached['signal'] == final_action: is_fresh = False

            if config.get("auto_trade_enabled") and is_fresh and trade_size > 0:
                side = "BUY" if "BUY" in final_action else "SELL"
                if not config.get("confirmation_required", True):
                    executed = self._execute_trade(
                        symbol, side, trade_size, trade_type, adjusted_score,
                        str(decision.reasons), decision.decision_id, config
                    )
                    if executed: decision.reasons.append(f"⚡ {trade_type}")

            self.signal_memory[symbol] = {"signal": final_action, "ts": time.time()}

            full_reason = "; ".join(decision.reasons)
            if penalties: full_reason += f" [{','.join(penalties)}]"

            # Dodajemy External Summary do powodu, jeśli istnieje
            if self.ai_core.external_context.get("summary"):
                 full_reason += f" | Ext: {self.ai_core.external_context['summary'][:15]}..."

            return {
                "symbol": symbol,
                "price": market_ctx["close"],
                "signal": final_action,
                "score": round(adjusted_score, 1),
                "raw_score": round(decision.score, 1),
                "quality": round(market_quality_scalar, 2),
                "size_rec": trade_type,
                "rsi": round(market_ctx["rsi"], 1),
                "confidence": decision.confidence,
                "reason": full_reason,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "is_new": is_fresh
            }

        except Exception as e:
            return None

    def run(self, config):
        """Główna pętla (Parallel)"""
        start_time = time.time()
        self.error_count = 0
        watchlist = self.get_dynamic_watchlist(limit=25)

        # 1. Market Quality
        market_quality = 1.0
        try:
            btc_df = DataFeed.get_market_data("BTC/USDT", "1h", limit=60)
            if not btc_df.empty:
                btc_ctx = self._build_context(btc_df)
                btc_dec = self.ai_core.evaluate("BTC/USDT", btc_ctx, btc_df, config)
                if btc_dec.score < 40: market_quality = 0.5
                elif btc_dec.score < 50: market_quality = 0.8
        except:
            market_quality = 0.8

        market_quality = max(0.25, market_quality)
        logger.info(f"🚀 Scanning {len(watchlist)} assets... Quality: {market_quality:.2f}")

        # 2. Multithreading
        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self._process_single_symbol, symbol, config, market_quality)
                for symbol in watchlist if symbol != "BTC/USDT"
            ]
            for future in futures:
                res = future.result()
                if res: results.append(res)
                else: self.error_count += 1

        results.sort(key=lambda x: abs(x['score'] - 50), reverse=True)
        logger.info(f"✅ Finished in {time.time() - start_time:.2f}s")

        return results
