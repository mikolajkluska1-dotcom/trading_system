"""
REDLINE SENTIMENT ENGINE
Simulates a high-frequency news feed for the Autonomous Node.
"""

import random
import time

class NewsAggregator:
    def __init__(self):
        self.last_fetch_time = 0
        self.sources = ["Twitter", "Bloomberg Terminal", "CoinDesk", "Whale Alert", "SEC.gov"]
        
        self.templates = [
            # Bullish
            ("Elon Musk tweets about {coin} (Bullish)", 0.85),
            ("BlackRock applies for {coin} ETF", 0.95),
            ("Federal Reserve hints at rate cuts", 0.70),
            ("{coin} breaks key resistance level", 0.65),
            ("MicroStrategy buys more BTC", 0.75),
            
            # Bearish
            ("SEC sues major exchange over {coin}", -0.90),
            ("China bans crypto mining (again)", -0.80),
            ("Inflations rises unexpectedly", -0.70),
            ("Whale dumps 5000 {coin} on Binance", -0.85),
            ("{coin} network stalled for 2 hours", -0.75),
            
            # Neutral / Noise
            ("Analysts predict volatility for {coin}", 0.10),
            ("{coin} volume consolidates", 0.05),
            ("New crypto regulation proposed in EU", -0.20),
            ("DeFi TVL reaches new monthly high", 0.40)
        ]
        
        self.coins = ["BTC", "ETH", "SOL", "DOGE", "XRP"]

    def fetch_latest_sentiment(self):
        """
        Simulate fetching breaking news.
        Returns None if no new news (random throttled), or a News Dictionary.
        """
        # 30% chance to find "news" every poll cycle
        if random.random() > 0.3:
            return None
            
        template, base_score = random.choice(self.templates)
        coin = random.choice(self.coins)
        
        # Add some random variance to the score
        final_score = base_score + random.uniform(-0.1, 0.1)
        final_score = max(-1.0, min(1.0, final_score)) # Clamp -1 to 1
        
        headline = template.format(coin=coin)
        
        return {
            "headline": headline,
            "sentiment_score": round(final_score, 2),
            "source": random.choice(self.sources),
            "timestamp": time.time(),
            "impact_level": "HIGH" if abs(final_score) > 0.7 else "MEDIUM" if abs(final_score) > 0.4 else "LOW"
        }
