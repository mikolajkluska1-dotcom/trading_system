import random
from textblob import TextBlob

class NewsDesk:
    """Moduł Analizy Sentymentu (NLP)"""
    
    @staticmethod
    def get_sentiment(symbol):
        # Symulowane nagłówki (dla celów demo/testu)
        headlines = [
            f"{symbol} partnership announced with major bank",
            f"Regulatory concerns regarding {symbol} market",
            f"{symbol} breaks all time high resistance level",
            "Global market fear increases due to inflation data",
            f"Analysts predict huge move for {symbol} next week",
            f"{symbol} volume spikes significantly",
            "Bearish divergence spotted on major timeframe"
        ]

        # Losujemy newsa
        news = random.choice(headlines)
        
        try:
            # Analiza sentymentu (-1.0 do 1.0)
            analysis = TextBlob(news)
            score = analysis.sentiment.polarity
        except:
            score = 0.0

        return score, news