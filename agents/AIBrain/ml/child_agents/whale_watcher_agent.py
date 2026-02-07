from .base_agent import BaseAgent
import pandas as pd

class WhaleWatcherAgent(BaseAgent):
    def __init__(self, name="whale_watcher"):
        super().__init__(name)

    async def analyze(self, market_data):
        df = market_data.get('klines')
        if df is None or len(df) < 30:
            return {'signal': 'HOLD', 'score': 0.0}
            
        # 1. Relative Volume
        curr_vol = df.iloc[-1]['volume']
        avg_vol = df['volume'].rolling(24).mean().iloc[-1]
        if avg_vol == 0: return {'signal': 'HOLD', 'score': 0.0}
        rvol = curr_vol / avg_vol
        
        # 2. Candle Analysis (Buying vs Selling Pressure)
        # Obliczamy, gdzie zamknęła się świeca względem jej zakresu (High-Low)
        high = df.iloc[-1]['high']
        low = df.iloc[-1]['low']
        close = df.iloc[-1]['close']
        open_p = df.iloc[-1]['open']
        
        range_len = high - low
        if range_len == 0: return {'signal': 'HOLD', 'score': 0.0}
        
        # Pozycja zamknięcia (0.0 = Low, 1.0 = High)
        close_pos = (close - low) / range_len
        
        score = 0.0
        signal = 'HOLD'
        reason = []
        
        # SCENARIUSZ A: PUMP (Duży wolumen + Zamknięcie wysoko)
        if rvol > 2.0 and close_pos > 0.8:
            score = 0.9
            signal = 'BUY'
            reason.append("High Vol Buying Pressure")
            
        # SCENARIUSZ B: REJECTION (Duży wolumen + Długi górny knot)
        elif rvol > 2.0 and close_pos < 0.2:
            score = 0.9
            signal = 'SELL'
            reason.append("High Vol Rejection (Wick)")
            
        # SCENARIUSZ C: CHURNING (Duży wolumen, małe ciało świecy - walka)
        elif rvol > 3.0 and abs(close - open_p) < (range_len * 0.3):
            score = 0.5
            signal = 'HOLD' # Niepewność, ale duża aktywność
            reason.append("Whale Fight (Indecision)")
            
        return {
            'signal': signal, 
            'score': score, 
            'reasoning': f"RVOL: {rvol:.1f}, {', '.join(reason)}"
        }
