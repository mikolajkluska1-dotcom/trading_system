import requests
import time
import logging

logger = logging.getLogger("SENTIMENT")

class SentimentOracle:
    """
    KRYPTOWALUTOWY ORACLE V1.
    Pobiera 'Fear & Greed Index' jako globalny bias sentymentu.
    """
    _cache = {"value": 50, "ts": 0}
    
    @staticmethod
    def get_fear_greed_index():
        # Cache 1h
        if time.time() - SentimentOracle._cache['ts'] < 3600:
            return SentimentOracle._cache['value']
            
        try:
            # Publiczne darmowe API (alternatywa)
            url = "https://api.alternative.me/fng/"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            val = int(data['data'][0]['value'])
            SentimentOracle._cache = {"value": val, "ts": time.time()}
            logger.info(f"🔮 Updated Sentiment: {val}/100")
            return val
        except Exception as e:
            logger.error(f"Sentment API Error: {e}")
            return SentimentOracle._cache['value'] # Return last known

    @staticmethod
    def get_sentiment_bias():
        """
        Zwraca mnożnik ryzyka (0.5 - 1.5) w zależności od nastroju.
        Contrarian Logic: 
        - Extreme Fear (<20) -> BUY Opportunity (Bias > 1.0)
        - Extreme Greed (>80) -> SELL Risk (Bias < 1.0)
        """
        val = SentimentOracle.get_fear_greed_index()
        
        # Logika Kontrariańska
        if val < 20: return 1.2, "EXTREME FEAR (Buy the blood)"
        if val < 40: return 1.1, "FEAR"
        if val > 80: return 0.8, "EXTREME GREED (Take profits)"
        if val > 60: return 0.9, "GREED"
        
        return 1.0, "NEUTRAL"
