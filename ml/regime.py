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
            
            score = 0
            # A. Czystosc
            if er > 0.6: score += 30
            elif er > 0.3: score += 15
            
            # B. Trend
            if 25 < adx < 60: score += 30
            elif adx >= 60: score += 15
            
            # C. Zmiennosc
            if 0.05 < bb_width < 0.30: score += 20
            
            # D. Momentum
            if 40 < rsi < 80: score += 20
            
            # Normalizacja (Safety Clamp)
            score = max(0, min(100, score))
            
            if score >= 80: regime = "SNIPER (Perfect Trend)"
            elif score >= 50: regime = "STANDARD (Tradeable)"
            else: regime = "TOXIC (Chop/Noise)"
            
            return score, regime

        except Exception as e:
            return 0, "DATA_ERROR"