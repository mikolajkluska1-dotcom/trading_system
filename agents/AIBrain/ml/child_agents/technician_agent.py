"""
AIBrain v4.0 - HUNTER UPDATE
Technician Agent - THE SNIPER
Strategia: RSI + Bollinger Bands Confluence
UWAGA: Bez pandas_ta - czyste numpy/pandas
"""
from .base_agent import BaseAgent
import numpy as np
import pandas as pd


def calculate_rsi(series, length=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, length=20, std=2.0):
    """Calculate Bollinger Bands"""
    sma = series.rolling(window=length).mean()
    std_dev = series.rolling(window=length).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower


class TechnicianAgent(BaseAgent):
    def __init__(self, name="technician"):
        super().__init__(name, specialty="technical_confluence")

    def _initialize_dna(self) -> dict:
        return {
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'rsi_trend_low': 55,
            'rsi_trend_high': 70,
            'bb_length': 20,
            'bb_std': 2.0
        }

    async def analyze(self, market_data):
        df = market_data.get('klines') or market_data.get('df')
        if df is None or len(df) < 25:
            self.last_analysis = {'signal': 'HOLD', 'score': 0.0, 'reasoning': 'Insufficient data'}
            return self.last_analysis
        
        df = df.copy()
        
        # RSI
        rsi = calculate_rsi(df['close'], 14).iloc[-1]
        
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(
            df['close'], 
            self.dna['bb_length'], 
            self.dna['bb_std']
        )
        
        lower_band = lower.iloc[-1]
        upper_band = upper.iloc[-1]
        close = df.iloc[-1]['close']
        
        score = 0.0
        signal = 'HOLD'
        reasons = []
        
        # CONFLUENCE LOGIC
        if rsi < self.dna['rsi_oversold'] and close <= lower_band * 1.01:
            score = 0.9
            signal = 'BUY'
            reasons.append(f"RSI Oversold ({rsi:.0f}) + BB Lower Touch")
            
        elif rsi > self.dna['rsi_overbought'] and close >= upper_band * 0.99:
            score = 0.9
            signal = 'SELL'
            reasons.append(f"RSI Overbought ({rsi:.0f}) + BB Upper Touch")
            
        elif rsi > self.dna['rsi_trend_low'] and rsi < self.dna['rsi_trend_high']:
            score = 0.3
            signal = 'BUY'
            reasons.append(f"Trend Zone ({rsi:.0f})")
            
        elif rsi < 45 and rsi > 30:
            score = 0.3
            signal = 'SELL'
            reasons.append(f"Bearish Zone ({rsi:.0f})")
            
        self.last_analysis = {
            'signal': signal, 
            'score': score,
            'reasoning': ", ".join(reasons) if reasons else "No confluence"
        }
        return self.last_analysis
