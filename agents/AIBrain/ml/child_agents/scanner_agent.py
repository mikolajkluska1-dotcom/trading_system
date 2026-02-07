"""
AIBrain v4.0 - HUNTER UPDATE
Scanner Agent - THE OPPORTUNITY HUNTER
Strategia: Trend (EMA Stack) + Momentum (ROC) + Siła (ADX) + Volume (RVOL)
UWAGA: Bez pandas_ta - czyste numpy/pandas
"""
from .base_agent import BaseAgent
import numpy as np
import pandas as pd


def calculate_ema(series, length):
    """Calculate EMA"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_rsi(series, length=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_roc(series, length=6):
    """Calculate Rate of Change"""
    return ((series - series.shift(length)) / series.shift(length)) * 100

def calculate_adx(df, length=14):
    """Calculate ADX (simplified)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(length).mean()
    plus_di = 100 * (plus_dm.rolling(length).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(length).mean() / (atr + 1e-10))
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(length).mean()
    
    return adx


class ScannerAgent(BaseAgent):
    def __init__(self, name="scanner"):
        super().__init__(name, specialty="opportunity_hunting")

    def _initialize_dna(self) -> dict:
        return {
            'adx_threshold': 25,
            'roc_threshold': 0.5,
            'rvol_threshold': 1.5,
            'rsi_overbought': 85
        }

    async def analyze(self, market_data):
        df = market_data.get('klines') or market_data.get('df')
        if df is None or len(df) < 50:
            self.last_analysis = {'signal': 'HOLD', 'score': 0.0, 'reasoning': 'Not enough data'}
            return self.last_analysis
        
        df = df.copy()
        
        # 1. EMA STACK
        ema_short = calculate_ema(df['close'], 9)
        ema_mid = calculate_ema(df['close'], 21)
        ema_long = calculate_ema(df['close'], 50)
        
        current_close = df.iloc[-1]['close']
        
        is_bull_trend = (current_close > ema_short.iloc[-1]) and \
                        (ema_short.iloc[-1] > ema_mid.iloc[-1]) and \
                        (ema_mid.iloc[-1] > ema_long.iloc[-1])
        
        # 2. ROC
        roc = calculate_roc(df['close'], 6).iloc[-1]
        
        # 3. ADX
        adx = calculate_adx(df, 14)
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
        # 4. RVOL
        vol_avg = df['volume'].rolling(20).mean().iloc[-1]
        vol_cur = df.iloc[-1]['volume']
        rvol = vol_cur / vol_avg if vol_avg > 0 else 0
        
        # SCORING
        score = 0.0
        signal = 'HOLD'
        reasons = []
        
        if is_bull_trend:
            score += 0.3
            reasons.append("Trend: Bullish")
            
            if current_adx > self.dna['adx_threshold']:
                score += 0.2
                reasons.append("ADX: Strong")
                
            if roc > self.dna['roc_threshold']:
                score += 0.3
                reasons.append("Momentum: High")
                
            if rvol > self.dna['rvol_threshold']:
                score += 0.2
                reasons.append("Vol: Pump")
                signal = 'BUY'
        
        elif current_close < ema_long.iloc[-1] and roc < -1.0:
            score = 0.8
            signal = 'SELL'
            reasons.append("Trend: Bearish Breakdown")

        if signal == 'HOLD' and score >= 0.6:
            signal = 'BUY'
        
        rsi = calculate_rsi(df['close'], 14).iloc[-1]
        if signal == 'BUY' and rsi > self.dna['rsi_overbought']:
            score -= 0.3 
            reasons.append("Risk: Overbought")

        self.last_analysis = {
            'signal': signal, 
            'score': min(max(score, 0.0), 1.0),
            'reasoning': ", ".join(reasons) if reasons else "No setup"
        }
        return self.last_analysis
