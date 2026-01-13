import numpy as np

class MarketRegime:
    """
    V2.5: Market Quality Score + Safety Clamp.
    """

    @staticmethod
    def calculate_efficiency_ratio(close_series, period=10):
        if len(close_series) < period + 1: return 0.0
        change = abs(close_series.iloc[-1] - close_series.iloc[-period - 1])
        volatility = np.sum(np.abs(close_series.diff().tail(period)))
        if volatility == 0: return 0.0
        return change / volatility

    @staticmethod
    def analyze(df):
        if df.empty or len(df) < 20:
            return 0, "NO_DATA"

        try:
            adx = df['adx'].iloc[-1] if 'adx' in df else 0
            bb_width = df['bb_width'].iloc[-1] if 'bb_width' in df else 0
            rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50
            er = MarketRegime.calculate_efficiency_ratio(df['close'], 10)

            # Nowość: Sprawdzanie akceleracji (Volume + Volatilty Expansion)
            vol_ma = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df and 'volume' in df.columns else 1
            curr_vol = df['volume'].iloc[-1] if 'volume' in df and 'volume' in df.columns else 1
            vol_spike = curr_vol / vol_ma if vol_ma > 0 else 1

            score = 0
            # A. Czystosc (Efficiency Ratio)
            if er > 0.6: score += 30
            elif er > 0.3: score += 15

            # B. Siła Trendu
            if 25 < adx < 60: score += 30
            elif adx >= 60: score += 15

            # C. Zmiennosc i Squeeze
            if 0.05 < bb_width < 0.30: score += 20
            elif bb_width < 0.05: score += 10 # Low volatility Squeeze

            # D. Momentum i Akceleracja
            if 40 < rsi < 80: score += 15
            if vol_spike > 2.0: score += 5

            # Normalizacja
            score = max(0, min(100, score))

            # --- Klasyfikacja Faz Rynku (Rich Context) ---
            if score >= 80:
                if rsi > 85 and vol_spike > 3.0:
                    regime = "BLOW-OFF (Exhaustion Risk)"
                else:
                    regime = "ACCELERATION (Strong Trend)"
            elif score >= 50:
                regime = "STANDARD (Tradeable)"
            elif score >= 30:
                regime = "ACCUMULATION (Ranging)"
            else:
                regime = "TOXIC (Chop/Noise)"

            return score, regime

        except Exception as e:
            return 0, f"DATA_ERROR: {str(e)}"
