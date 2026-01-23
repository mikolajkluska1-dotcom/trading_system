import ccxt
import logging
import time

logger = logging.getLogger("MICROSTRUCTURE")

class OrderBookAnalyzer:
    """
    ANALIZATOR MIKROSTRUKTURY RYNKU (OBI)
    Analizuje głębokość rynku (Depth) w poszukiwaniu 'ścian' (Walls).
    """
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        
    def get_imbalance(self, symbol, depth=20):
        """
        Oblicza Order Book Imbalance (OBI).
        > 1.0 - Przewaga Kupujących (Buy Wall)
        < 1.0 - Przewaga Sprzedających (Sell Wall)
        """
        try:
            # Pobieramy Order Book
            book = self.exchange.fetch_order_book(symbol, limit=depth)
            
            bids = book['bids']
            asks = book['asks']
            
            if not bids or not asks: return 1.0, "NO_DATA"

            # Sumujemy wolumeny w top N ofertach
            bid_vol = sum([x[1] for x in bids])
            ask_vol = sum([x[1] for x in asks])
            
            if ask_vol == 0: return 1.0, "ERR"
            
            ratio = bid_vol / ask_vol
            
            # Interpretacja
            label = "NEUTRAL"
            if ratio > 2.0: label = "STRONG_BUY_WALL"
            elif ratio > 1.3: label = "BUY_PRESSURE"
            elif ratio < 0.5: label = "STRONG_SELL_WALL"
            elif ratio < 0.7: label = "SELL_PRESSURE"
            
            logger.info(f"🧱 OBI {symbol}: Ratio {ratio:.2f} ({label})")
            return ratio, label
            
        except Exception as e:
            logger.error(f"OBI Analysis Failed: {e}")
            return 1.0, "ERR"
