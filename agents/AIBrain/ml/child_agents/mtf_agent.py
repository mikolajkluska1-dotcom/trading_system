"""
AIBrain v4.0 - MEGA UPGRADE
Multi-Timeframe Agent - THE CONFLUENCE MASTER
Strategia: Analiza 3 timeframe'ów (15m, 1h, 4h)
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

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic(df, k=14, d=3):
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=k).min()
    high_max = df['high'].rolling(window=k).max()
    
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    stoch_d = stoch_k.rolling(window=d).mean()
    
    return stoch_k, stoch_d


class MTFAgent(BaseAgent):
    def __init__(self, name="mtf"):
        super().__init__(name, specialty="multi_timeframe_confluence")
        self._tf_cache = {'15m': None, '1h': None, '4h': None}

    def _initialize_dna(self) -> dict:
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'ema_fast': 12,
            'ema_slow': 26,
            'weight_15m': 0.25,
            'weight_1h': 0.35,
            'weight_4h': 0.40
        }

    def set_timeframe_data(self, tf: str, df: pd.DataFrame):
        self._tf_cache[tf] = df

    def _analyze_15m(self, df) -> dict:
        if df is None or len(df) < 20:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': 'No 15m data'}
        
        try:
            rsi = calculate_rsi(df['close'], 14).iloc[-1]
            stoch_k, _ = calculate_stochastic(df, 14, 3)
            stoch = stoch_k.iloc[-1]
            
            if rsi < self.dna['rsi_oversold'] and stoch < 20:
                return {'signal': 'BUY', 'score': 0.9, 'reason': f'15m: Oversold'}
            elif rsi > self.dna['rsi_overbought'] and stoch > 80:
                return {'signal': 'SELL', 'score': 0.9, 'reason': f'15m: Overbought'}
            elif rsi < 45:
                return {'signal': 'BUY', 'score': 0.3, 'reason': '15m: Bullish'}
            elif rsi > 55:
                return {'signal': 'SELL', 'score': 0.3, 'reason': '15m: Bearish'}
            else:
                return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': '15m: Neutral'}
        except:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': '15m: Error'}

    def _analyze_1h(self, df) -> dict:
        if df is None or len(df) < 30:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': 'No 1h data'}
        
        try:
            ema_fast = calculate_ema(df['close'], self.dna['ema_fast'])
            ema_slow = calculate_ema(df['close'], self.dna['ema_slow'])
            _, _, macd_hist = calculate_macd(df['close'])
            
            close = df.iloc[-1]['close']
            
            if close > ema_fast.iloc[-1] > ema_slow.iloc[-1] and macd_hist.iloc[-1] > 0:
                return {'signal': 'BUY', 'score': 0.8, 'reason': '1h: Bull + MACD'}
            elif close < ema_fast.iloc[-1] < ema_slow.iloc[-1] and macd_hist.iloc[-1] < 0:
                return {'signal': 'SELL', 'score': 0.8, 'reason': '1h: Bear + MACD'}
            elif close > ema_slow.iloc[-1]:
                return {'signal': 'BUY', 'score': 0.4, 'reason': '1h: Above EMA'}
            elif close < ema_slow.iloc[-1]:
                return {'signal': 'SELL', 'score': 0.4, 'reason': '1h: Below EMA'}
            else:
                return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': '1h: Mixed'}
        except:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': '1h: Error'}

    def _analyze_4h(self, df) -> dict:
        if df is None or len(df) < 50:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': 'No 4h data'}
        
        try:
            ema_50 = calculate_ema(df['close'], 50)
            close = df.iloc[-1]['close']
            
            if close > ema_50.iloc[-1]:
                return {'signal': 'BUY', 'score': 0.6, 'reason': '4h: Above EMA50'}
            else:
                return {'signal': 'SELL', 'score': 0.6, 'reason': '4h: Below EMA50'}
        except:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'reason': '4h: Error'}

    async def analyze(self, market_data):
        df_15m = market_data.get('df_15m') or self._tf_cache.get('15m')
        df_1h = market_data.get('df_1h') or market_data.get('klines') or market_data.get('df')
        df_4h = market_data.get('df_4h') or self._tf_cache.get('4h')
        
        result_15m = self._analyze_15m(df_15m)
        result_1h = self._analyze_1h(df_1h)
        result_4h = self._analyze_4h(df_4h)
        
        signals = [result_15m['signal'], result_1h['signal'], result_4h['signal']]
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        
        reasons = [r['reason'] for r in [result_15m, result_1h, result_4h] if r['signal'] != 'NEUTRAL']
        
        if buy_votes == 3:
            signal, score, confluence = 'BUY', 0.95, 'STRONG'
        elif sell_votes == 3:
            signal, score, confluence = 'SELL', 0.95, 'STRONG'
        elif buy_votes >= 2 and sell_votes == 0:
            signal, score, confluence = 'BUY', 0.6, 'MODERATE'
        elif sell_votes >= 2 and buy_votes == 0:
            signal, score, confluence = 'SELL', 0.6, 'MODERATE'
        else:
            signal, score, confluence = 'HOLD', 0.2, 'WEAK'
        
        self.last_analysis = {
            'signal': signal,
            'score': score,
            'reasoning': " | ".join(reasons) if reasons else "No confluence",
            'confluence': confluence,
            'votes': {'buy': buy_votes, 'sell': sell_votes}
        }
        return self.last_analysis

    def get_signal_for_attention(self) -> float:
        score = self.last_analysis.get('score', 0.0)
        signal = self.last_analysis.get('signal', 'HOLD')
        return -score if signal == 'SELL' else (score if signal == 'BUY' else 0.0)
