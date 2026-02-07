"""
AIBrain v4.0 - MEGA UPGRADE
Trend Agent - THE MACRO NAVIGATOR
Strategia: SuperTrend + EMA200 dla macro trend detection
UWAGA: Bez pandas_ta - czyste numpy/pandas
"""
from .base_agent import BaseAgent
import numpy as np
import pandas as pd


def calculate_ema(series, length):
    """Calculate EMA"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_atr(df, length=10):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    return tr.rolling(length).mean()

def calculate_supertrend(df, length=10, multiplier=3.0):
    """Calculate SuperTrend"""
    atr = calculate_atr(df, length)
    hl2 = (df['high'] + df['low']) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    close = df['close']
    
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1  # Bullish
            supertrend.iloc[i] = lower_band.iloc[i]
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1  # Bearish
            supertrend.iloc[i] = upper_band.iloc[i]
        else:
            direction.iloc[i] = direction.iloc[i-1] if i > 0 else 1
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1]) if i > 0 else lower_band.iloc[i]
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1]) if i > 0 else upper_band.iloc[i]
    
    return supertrend, direction


class TrendAgent(BaseAgent):
    def __init__(self, name="trend"):
        super().__init__(name, specialty="macro_trend_detection")

    def _initialize_dna(self) -> dict:
        return {
            'supertrend_length': 10,
            'supertrend_multiplier': 3.0,
            'ema_long': 200,
            'ema_medium': 50
        }

    async def analyze(self, market_data):
        df = market_data.get('klines') or market_data.get('df')
        
        if df is None or len(df) < 200:
            self.last_analysis = {
                'signal': 'HOLD', 
                'score': 0.0, 
                'reasoning': 'Need 200+ candles',
                'trend': 'UNKNOWN'
            }
            return self.last_analysis
        
        df = df.copy()
        close = df.iloc[-1]['close']
        trend_votes = {'bull': 0, 'bear': 0}
        reasons = []
        
        # 1. SUPERTREND
        try:
            st, st_dir = calculate_supertrend(
                df, 
                self.dna['supertrend_length'],
                self.dna['supertrend_multiplier']
            )
            
            if st_dir.iloc[-1] == 1:
                trend_votes['bull'] += 1
                reasons.append("ST: Bull")
            elif st_dir.iloc[-1] == -1:
                trend_votes['bear'] += 1
                reasons.append("ST: Bear")
        except:
            pass
        
        # 2. EMA 200
        try:
            ema_200 = calculate_ema(df['close'], self.dna['ema_long'])
            ema_50 = calculate_ema(df['close'], self.dna['ema_medium'])
            
            if close > ema_200.iloc[-1]:
                trend_votes['bull'] += 1
                if ema_50.iloc[-1] > ema_200.iloc[-1]:
                    reasons.append("EMA: Golden Cross")
                else:
                    reasons.append("EMA: Above 200")
            else:
                trend_votes['bear'] += 1
                reasons.append("EMA: Below 200")
        except:
            pass
        
        # DECISION
        if trend_votes['bull'] >= 2:
            trend = 'UPTREND'
            signal = 'BUY'
            score = 0.7
        elif trend_votes['bear'] >= 2:
            trend = 'DOWNTREND'
            signal = 'SELL'
            score = 0.7
        elif trend_votes['bull'] > trend_votes['bear']:
            trend = 'UPTREND'
            signal = 'BUY'
            score = 0.4
        elif trend_votes['bear'] > trend_votes['bull']:
            trend = 'DOWNTREND'
            signal = 'SELL'
            score = 0.4
        else:
            trend = 'SIDEWAYS'
            signal = 'HOLD'
            score = 0.2
        
        self.last_analysis = {
            'signal': signal,
            'score': score,
            'reasoning': ", ".join(reasons) if reasons else "No clear trend",
            'trend': trend,
            'trend_votes': trend_votes
        }
        return self.last_analysis

    def get_signal_for_attention(self) -> float:
        score = self.last_analysis.get('score', 0.0)
        signal = self.last_analysis.get('signal', 'HOLD')
        
        if signal == 'SELL':
            return -score
        elif signal == 'BUY':
            return score
        return 0.0
