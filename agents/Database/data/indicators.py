# ================================================================
# REDLINE V68 - TECHNICAL INDICATORS MODULE (V4.6-Pro)
# Author: Quant Research Division
# Description:
#   Kompletny zestaw wskaźników technicznych używanych przez
#   moduł DeepBrain (AI), MarketRegime oraz DataFeed.
# ================================================================

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class TechnicalIndicators:
    """
    Moduł Analityczny V4.6-Pro
    Oblicza zestaw wskaźników technicznych wymaganych przez system AI.
    Wymaga kolumn: ['high', 'low', 'close', 'v'] lub ['volume'].
    """

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """Aplikuje komplet wskaźników technicznych (TA)."""
        if df is None or df.empty:
            return df

        try:
            data = df.copy()

            # Wybór kolumny wolumenu
            vol_col = "v" if "v" in data.columns else "volume"

            # ============================================================
            # 1️⃣ RSI (Momentum)
            # ============================================================
            rsi = RSIIndicator(close=data["close"], window=14, fillna=True)
            data["rsi"] = rsi.rsi()

            # ============================================================
            # 2️⃣ MACD (Trend)
            # ============================================================
            macd = MACD(
                close=data["close"],
                window_slow=26,
                window_fast=12,
                window_sign=9,
                fillna=True
            )
            data["macd"] = macd.macd()
            data["macd_diff"] = macd.macd_diff()
            data["macd_signal"] = macd.macd_signal()

            # ============================================================
            # 3️⃣ Bollinger Bands (Zmienność)
            # ============================================================
            bb = BollingerBands(close=data["close"], window=20, window_dev=2, fillna=True)
            data["bb_high"] = bb.bollinger_hband()
            data["bb_low"] = bb.bollinger_lband()
            data["bb_width"] = bb.bollinger_wband()

            # ============================================================
            # 4️⃣ ATR (Zmienność absolutna)
            # ============================================================
            atr = AverageTrueRange(
                high=data["high"],
                low=data["low"],
                close=data["close"],
                window=14,
                fillna=True
            )
            data["atr"] = atr.average_true_range()

            # ============================================================
            # 5️⃣ ADX (Siła trendu)
            # ============================================================
            adx = ADXIndicator(high=data["high"], low=data["low"], close=data["close"], window=14, fillna=True)
            data["adx"] = adx.adx()
            # Dodatkowe składniki DI
            data["di_pos"] = adx.adx_pos()
            data["di_neg"] = adx.adx_neg()

            # ============================================================
            # 6️⃣ EMA (Średnie kroczące)
            # ============================================================
            data["ema_50"] = EMAIndicator(close=data["close"], window=50, fillna=True).ema_indicator()
            data["ema_200"] = EMAIndicator(close=data["close"], window=200, fillna=True).ema_indicator()

            # ============================================================
            # 7️⃣ Log Returns (Zwroty logarytmiczne)
            # ============================================================
            data["ret"] = np.log(data["close"] / data["close"].shift(1))

            # ============================================================
            # 8️⃣ Sanity Filter / Safety Layer
            # ============================================================
            required_cols = [
                "rsi", "macd", "macd_diff", "macd_signal",
                "bb_high", "bb_low", "bb_width",
                "atr", "adx", "di_pos", "di_neg",
                "ema_50", "ema_200", "ret"
            ]
            for c in required_cols:
                if c not in data.columns:
                    data[c] = 0.0

            # Usuwamy NaN i nieskończoności
            data.replace([np.inf, -np.inf], 0, inplace=True)
            data.fillna(0, inplace=True)

            return data

        except Exception as e:
            print(f"[Indicators] Error calculating indicators: {e}")
            return df
