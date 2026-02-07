"""
AIBrain v3.0 - Sentiment Agent
Market sentiment analysis using Fear & Greed Index
"""
from .base_agent import BaseAgent
import logging
import requests
import time
import os
import pandas as pd

logger = logging.getLogger("SENTIMENT_AGENT")


class SentimentAgent(BaseAgent):
    """
    Sentiment Agent - Market Sentiment Analysis
    
    Uses Fear & Greed Index and other sentiment data
    with contrarian logic (buy fear, sell greed).
    """
    
    _cache = {"value": 50, "ts": 0}
    
    def __init__(self, name="sentiment"):
        super().__init__(name, specialty="sentiment_analysis")
        self.fear_greed_path = "R:/Redline_Data/sentiment/fear_greed.csv"
    
    def _initialize_dna(self) -> dict:
        """DNA: Sentiment thresholds"""
        return {
            'extreme_fear': 20,
            'fear': 35,
            'greed': 65,
            'extreme_greed': 80,
            'contrarian_weight': 1.0
        }
    
    def _get_fear_greed(self) -> int:
        """Get Fear & Greed Index (cached 1h)"""
        
        # Check cache
        if time.time() - self._cache['ts'] < 3600:
            return self._cache['value']
        
        # Try local file first
        if os.path.exists(self.fear_greed_path):
            try:
                df = pd.read_csv(self.fear_greed_path)
                if len(df) > 0:
                    val = int(df.iloc[0]['value'])
                    self._cache = {"value": val, "ts": time.time()}
                    return val
            except:
                pass
        
        # Fallback to API
        try:
            url = "https://api.alternative.me/fng/"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            val = int(data['data'][0]['value'])
            self._cache = {"value": val, "ts": time.time()}
            logger.info(f"🔮 Updated Fear & Greed: {val}/100")
            return val
        except Exception as e:
            logger.error(f"Sentiment API Error: {e}")
            return self._cache['value']
    
    async def analyze(self, market_data: dict) -> dict:
        """
        Analyze market sentiment
        
        Uses contrarian logic:
        - Extreme Fear (<20) -> BUY opportunity
        - Extreme Greed (>80) -> SELL signal
        
        Args:
            market_data: dict (not used for global sentiment)
        
        Returns:
            dict with signal, score, confidence, reasoning
        """
        try:
            fg_index = self._get_fear_greed()
            
            # Contrarian logic
            if fg_index < 20:
                signal = 'BUY'
                score = 0.9
                reasoning = f"EXTREME FEAR ({fg_index}) - Buy the blood!"
                confidence = 0.85
            elif fg_index < 35:
                signal = 'BUY'
                score = 0.6
                reasoning = f"Fear ({fg_index}) - Contrarian buy signal"
                confidence = 0.7
            elif fg_index > 80:
                signal = 'SELL'
                score = -0.9
                reasoning = f"EXTREME GREED ({fg_index}) - Take profits!"
                confidence = 0.85
            elif fg_index > 65:
                signal = 'SELL'
                score = -0.6
                reasoning = f"Greed ({fg_index}) - Contrarian sell signal"
                confidence = 0.7
            else:
                signal = 'NEUTRAL'
                score = (50 - fg_index) / 100  # Slight bias
                reasoning = f"Neutral sentiment ({fg_index})"
                confidence = 0.5
            
            self.last_analysis = {
                'signal': signal,
                'score': score,
                'confidence': confidence,
                'reasoning': reasoning,
                'fear_greed_index': fg_index
            }
            
            return self.last_analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            self.last_analysis = {'signal': 'NEUTRAL', 'score': 0.0}
            return self.last_analysis
    
    def get_signal_for_attention(self) -> float:
        """Return normalized score for Attention mechanism (-1 to 1)"""
        return self.last_analysis.get('score', 0.0)
