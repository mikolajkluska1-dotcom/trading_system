import ccxt
import pandas as pd
import numpy as np
import os
from datetime import datetime

# CORE IMPORTS
from data.feed import DataFeed
from ml.brain import DeepBrain
from ml.regime import MarketRegime
from data.indicators import TechnicalIndicators
from core.event_logger import EventLogger  # <--- NOWOŚĆ

class MarketScanner:
    """
    REDLINE Market Scanner V6 (Ops-Ready).
    Zintegrowany z EventLoggerem: Generuje SIGNAL_ID dla każdego setupu.
    """

    def __init__(self, timeframe="1h"):
        self.tf = timeframe
        self.htf = "4h"
        self.brain = DeepBrain()
        self.output_path = os.path.join("assets", "scan_log.csv")

    def get_top_volume_symbols(self, limit=50):
        try:
            ex = ccxt.binance()
            tickers = ex.fetch_tickers()
            valid_tickers = [
                data for symbol, data in tickers.items() 
                if "/USDT" in symbol and "UP/" not in symbol and "DOWN/" not in symbol
            ]
            sorted_tickers = sorted(valid_tickers, key=lambda x: x['quoteVolume'], reverse=True)
            return [t['symbol'] for t in sorted_tickers[:limit]]
        except:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

    def _get_htf_trend(self, symbol):
        try:
            df = DataFeed.get_market_data(symbol, self.htf, limit=100)
            if df.empty: return "NEUTRAL"
            if 'ema_50' not in df.columns: df = TechnicalIndicators.add_all(df)
            return "BULLISH" if df['close'].iloc[-1] > df['ema_50'].iloc[-1] else "BEARISH"
        except:
            return "NEUTRAL"

    def scan(self):
        results = []
        symbols = self.get_top_volume_symbols(limit=50)
        
        print(f"--- OPS SCAN START: {len(symbols)} Assets ---")
        
        for i, symbol in enumerate(symbols):
            try:
                # 1. Dane LTF
                df = DataFeed.get_market_data(symbol, self.tf, limit=100)
                if df.empty or len(df) < 60: continue
                
                # 2. Regime
                mqs, _ = MarketRegime.analyze(df)
                if mqs < 35: continue

                # 3. AI Prediction
                pred_price, conf, signal = self.brain.predict(df)
                if conf < 0.60: continue

                # 4. HTF Context
                htf_trend = self._get_htf_trend(symbol)
                if signal == "BUY" and htf_trend == "BEARISH" and mqs < 80:
                    continue
                
                # 5. EV Calculation
                last_price = df["close"].iloc[-1]
                growth_pct = (pred_price - last_price) / last_price * 100
                
                atr = df['atr'].iloc[-1] if 'atr' in df else (last_price * 0.01)
                vol_pct = (atr / last_price) * 100 or 1.0
                rr = growth_pct / vol_pct
                
                # EV Score (Uproszczony na potrzeby logowania)
                ev_val = (growth_pct * conf) - (abs(growth_pct) * (1 - conf))
                
                final_score = (conf ** 1.5) * rr * 100

                # 6. LOGOWANIE SYGNAŁU (EventLogger)
                # Generujemy unikalny ID dla tego pomysłu
                signal_id = EventLogger.log_signal(
                    symbol=symbol,
                    tf=self.tf,
                    signal=signal,
                    conf=conf,
                    mqs=mqs,
                    htf_trend=htf_trend,
                    ev=ev_val,
                    reasons=["AI_HIT", "MQS_OK", f"SCORE_{final_score:.1f}"]
                )

                results.append({
                    "signal_id": signal_id, # <--- KLUCZOWE: Przekazujemy ID dalej
                    "symbol": symbol,
                    "signal": signal,
                    "conf": round(conf, 2),
                    "mqs": mqs,
                    "score": round(final_score, 2),
                    "current_price": last_price,
                    "ev": ev_val,
                    "ts": datetime.now().strftime("%H:%M")
                })
                
                print(f"[{i+1}] {symbol}: HIT! ID: {signal_id}")

            except Exception as e:
                print(f"Error {symbol}: {e}")
                continue

        if results:
            ranked = pd.DataFrame(results).sort_values("score", ascending=False)
            return ranked
        else:
            return pd.DataFrame()