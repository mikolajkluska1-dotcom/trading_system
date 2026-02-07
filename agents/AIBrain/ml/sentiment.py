"""
AIBrain v3.0 - Sentiment Oracle (v3 Compatible)
================================================
Pobiera 'Fear & Greed Index' jako globalny bias sentymentu.
Now with v3 DNA and attention-compatible methods.
"""
import requests
import time
import logging

logger = logging.getLogger("SENTIMENT")


class SentimentOracle:
    """
    CRYPTO SENTIMENT ORACLE v3.0
    Uses Fear & Greed Index with contrarian logic.
    """
    _cache = {"value": 50, "ts": 0}
    
    def __init__(self):
        # v3 DNA
        self.dna = {
            'extreme_fear': 20,
            'fear': 35,
            'greed': 65,
            'extreme_greed': 80,
            'contrarian_weight': 1.0
        }
        
        # v3 state for attention mechanism
        self.last_analysis = {'signal': 'NEUTRAL', 'score': 0.0}
    
    @staticmethod
    def get_fear_greed_index():
        """Fetch Fear & Greed Index from API (cached 1h)"""
        if time.time() - SentimentOracle._cache['ts'] < 3600:
            return SentimentOracle._cache['value']
            
        try:
            url = "https://api.alternative.me/fng/"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            val = int(data['data'][0]['value'])
            SentimentOracle._cache = {"value": val, "ts": time.time()}
            logger.info(f"🔮 Updated Sentiment: {val}/100")
            return val
        except Exception as e:
            logger.error(f"Sentiment API Error: {e}")
            return SentimentOracle._cache['value']

    @staticmethod
    def get_sentiment_bias():
        """
        Returns risk multiplier (0.5 - 1.5) based on sentiment.
        Contrarian Logic: 
        - Extreme Fear (<20) -> BUY Opportunity (Bias > 1.0)
        - Extreme Greed (>80) -> SELL Risk (Bias < 1.0)
        """
        val = SentimentOracle.get_fear_greed_index()
        
        if val < 20: return 1.2, "EXTREME FEAR (Buy the blood)"
        if val < 40: return 1.1, "FEAR"
        if val > 80: return 0.8, "EXTREME GREED (Take profits)"
        if val > 60: return 0.9, "GREED"
        
        return 1.0, "NEUTRAL"
    
    # ========== v3 Methods ==========
    
    async def analyze(self, market_data):
        """
        v3 Compatible analyze method for Mother Brain.
        Returns contrarian signal based on Fear & Greed.
        """
        fng = self.get_fear_greed_index()
        
        extreme_fear = self.dna.get('extreme_fear', 20)
        fear = self.dna.get('fear', 35)
        greed = self.dna.get('greed', 65)
        extreme_greed = self.dna.get('extreme_greed', 80)
        weight = self.dna.get('contrarian_weight', 1.0)
        
        # Contrarian logic: Buy fear, sell greed
        if fng < extreme_fear:
            score = 0.6 * weight  # Strong BUY
            signal = 'BUY'
            reason = f'Extreme Fear ({fng}) - Contrarian BUY'
        elif fng < fear:
            score = 0.3 * weight
            signal = 'BUY'
            reason = f'Fear ({fng}) - Mild BUY'
        elif fng > extreme_greed:
            score = -0.6 * weight  # Strong SELL
            signal = 'SELL'
            reason = f'Extreme Greed ({fng}) - Contrarian SELL'
        elif fng > greed:
            score = -0.3 * weight
            signal = 'SELL'
            reason = f'Greed ({fng}) - Mild SELL'
        else:
            score = 0.0
            signal = 'NEUTRAL'
            reason = f'Neutral sentiment ({fng})'
        
        self.last_analysis = {
            'signal': signal,
            'score': score,
            'reasoning': reason,
            'fear_greed_index': fng
        }
        return self.last_analysis
    
    def get_signal_for_attention(self) -> float:
        """v3: Return score for Attention mechanism"""
        return self.last_analysis.get('score', 0.0)
