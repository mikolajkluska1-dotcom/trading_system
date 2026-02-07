from .base_agent import BaseAgent
import pandas as pd

class PortfolioManagerAgent(BaseAgent):
    def __init__(self, name="portfolio_manager"):
        super().__init__(name)

    async def analyze(self, market_data):
        df = market_data.get('klines')
        if df is None or len(df) < 20:
            return {'signal': 'HOLD', 'score': 0.0}
            
        # Oblicz ATR (Zmienność) - manualnie bez pandas_ta
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        close = df.iloc[-1]['close']
        
        # Oblicz relatywną zmienność w %
        volatility_pct = (atr / close) * 100
        
        # Decyzja o wielkości pozycji (Risk Management)
        score = 0.0
        signal = 'HOLD'
        reason = []
        
        # Jeśli zmienność jest ekstremalna (>5% na świeczkę), zmniejszamy ryzyko
        if volatility_pct > 5.0:
            score = -0.5 # Sygnał negatywny dla wielkości pozycji
            signal = 'RISK_OFF' # Zmniejsz pozycję
            reason.append("Extreme Volatility")
        elif volatility_pct < 1.0:
            score = 0.8
            signal = 'RISK_ON' # Można grać agresywniej
            reason.append("Stable Market")
            
        # Output zawiera sugerowany Stop Loss
        suggested_sl = atr * 2.0 # 2x ATR to standardowy bezpieczny SL
        
        return {
            'signal': signal, 
            'score': score, 
            'volatility': volatility_pct,
            'suggested_sl_price': close - suggested_sl,
            'reasoning': f"Vol: {volatility_pct:.2f}%"
        }
