from .base_agent import BaseAgent
import pandas as pd

class CorrelationAgent(BaseAgent):
    def __init__(self, name="correlation_guard"):
        super().__init__(name)
        self.btc_trend = 0 # 1 = Bull, -1 = Bear

    async def analyze(self, market_data):
        # Ten agent w systemie live powinien mieć dostęp do danych BTC
        # W symulacji zakładamy, że jeśli analizowany symbol to NIE jest BTC,
        # to powinniśmy sprawdzić ogólny sentyment (tutaj uproszczone proxy).
        
        df = market_data.get('klines')
        if df is None or len(df) < 50:
            return {'signal': 'HOLD', 'score': 0.0}
            
        # Obliczamy szybki trend na obecnym coinie (EMA 50 manualnie)
        ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        close = df.iloc[-1]['close']
        
        # Obliczamy korelację z rynkiem (uproszczona: "Czy mocno spadamy?")
        # Jeśli cena jest > 5% pod EMA50, to jest crash, nie łapiemy spadających noży
        dist_to_ema = (close - ema50) / ema50
        
        score = 0.0
        signal = 'HOLD'
        reason = []
        
        # SAFETY CHECK
        if dist_to_ema < -0.05: # -5% od średniej
            signal = 'VETO' # Blokada zakupów
            score = -1.0
            reason.append("Market Crash Detected")
        elif dist_to_ema > 0.02:
            signal = 'BUY' # Bezpiecznie
            score = 0.5
            reason.append("Market Healthy")
            
        return {
            'signal': signal, 
            'score': score, 
            'reasoning': ", ".join(reason)
        }
