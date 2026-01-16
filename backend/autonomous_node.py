import time
import requests
import threading
import logging
import json
from datetime import datetime

# Component Imports
from backend.ai_core import RedlineAICore
from backend.data.news import NewsAggregator
from backend.google_reporting import GoogleReporter  # <--- NEW IMPORT
from ml.scanner import MarketScanner
from ml.training import OfflineTrainer
from ml.evolution import GeneticEvolution
from trading.wallet import WalletManager

# Logging
logger = logging.getLogger("AUTONOMOUS_NODE")
logger.setLevel(logging.INFO)

class RedlineAutonomousNode:
    """
    THE OVERSEER.
    Central Nervous System of the Redline Organism.
    Manages the infinite loop of: Scanning -> Reasoning -> Evolving.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedlineAutonomousNode, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized: return
        
        logger.info("🤖 Initializing Autonomous Node...")
        self.running = False
        self.status = "OFFLINE"
        
        # Core Subsystems
        self.ai_core = RedlineAICore()
        self.scanner = MarketScanner(self.ai_core)
        self.trainer = OfflineTrainer()
        self.news_engine = NewsAggregator()
        self.reporter = GoogleReporter() # <--- NEW REPORTER
        
        # State
        self.latest_scan_results = []
        self.last_evolution_time = 0
        self.evolution_interval = 86400 # 24h in seconds
        
        # Config (can be loaded from external file)
        self.config = {
            "min_confidence": 0.6,
            "auto_trade_enabled": False, # Safer default
            "execution_enabled": False
        }

        self.initialized = True

    def start(self):
        """Ignites the Spark of Life."""
        if self.running:
            logger.warning("🤖 Autonomous Node is already running.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._life_cycle, daemon=True)
        self.thread.start()
        logger.info("🤖 Autonomous Node STARTED. Entering Life Cycle.")

    def stop(self):
        self.running = False
        self.status = "SHUTTING_DOWN"
        logger.info("🤖 Autonomous Node STOPPING...")

    def _broadcast_update(self, msg_type, payload):
        """Sends data to the API Server to be broadcasted via WebSockets."""
        try:
            url = "http://localhost:8000/api/webhook/internal"
            requests.post(url, json={"type": msg_type, "payload": payload}, timeout=1)
        except Exception as e:
            # Silence connection errors (API might be down), log others
            pass

    def _broadcast_wallet_state(self):
        """Broadcasting wallet state updates to the Frontend."""
        try:
            data = WalletManager.get_wallet_data()
            payload = {
                "total_balance": data.get("balance", 0.0),
                "pnl_percent": 0.0,  # Placeholder, needs history logic
                "assets": data.get("assets", []),
                "recent_transactions": data.get("history", [])[:10]
            }
            self._broadcast_update("WALLET_UPDATE", payload)
        except Exception as e:
            logger.error(f"Failed to broadcast wallet state: {e}")

    def execute_trade(self, symbol, action, amount, price, profit=0.0):
        """
        Executes a trade and logs it to Google Sheets.
        """
        logger.info(f"⚡ EXECUTING TRADE: {action} {symbol} @ ${price}")
        
        # 1. Update Internal Wallet (Mock/Simulated)
        # In a real scenario, this would call Binance API or WalletManager
        # WalletManager.update_balance(...) 
        
        # 2. LOG TO GOOGLE SHEETS
        self.reporter.log_trade(
            symbol=symbol,
            action=action,
            price=price,
            amount=amount,
            profit=profit
        )
        
        # 3. Broadcast Event
        self._broadcast_update("TRADE_EXECUTED", {
            "symbol": symbol,
            "action": action,
            "price": price,
            "amount": amount
        })

    def _life_cycle(self):
        """The Infinite Loop of the Organism."""
        while self.running:
            try:
                # --- PHASE 0: SENSORY INPUT (News/Sentiment) ---
                news_item = self.news_engine.fetch_latest_sentiment()
                if news_item:
                    score = news_item['sentiment_score']
                    logger.info(f"📰 {news_item['source']}: {news_item['headline']} (Score: {score})")
                    
                    # React to Sentiment
                    if score > 0.8:
                        logger.info(f"[BRAIN] Positive News Detected! Adjusting aggression parameters.")
                    elif score < -0.8:
                        logger.info(f"[BRAIN] FUD Detected. Tightening Stop-Loss.")

                    # Broadcast to Frontend HUD/Logs
                    self._broadcast_update("NEWS_UPDATE", news_item)


                # --- PHASE A: SCANNING (Senses) ---
                self.status = "SCANNING"
                # Reduce log noise, only log every 10th cycle or significant event? 
                # Keeping it verbose for demo as requested.
                # logger.info(f"👁️ Phase A: Scanning Markets... [{datetime.now().strftime('%H:%M:%S')}]")
                
                # Run Scanner
                results = self.scanner.run(self.config)
                if results:
                    self.latest_scan_results = results
                    # logger.info(f"👁️ Scan Complete. Found {len(results)} opportunities.")
                    self._broadcast_update("SCAN_COMPLETE", results)
                    
                    # AUTO TRADING CHECK (Demo Logic)
                    if self.config.get("execution_enabled"):
                        for res in results:
                            if res['confidence'] > 0.9: # High conviction
                                self.execute_trade(res['symbol'], res['signal'], 0.1, res['price'])

                else:
                    # logger.info("👁️ Scan Complete. No opportunities found.")
                    pass

                # --- PHASE B: EVOLUTION (Self-Improvement) ---
                current_time = time.time()
                if current_time - self.last_evolution_time > self.evolution_interval:
                    self.status = "EVOLVING"
                    logger.info("🧬 Phase B: Evolutionary Checkpoint Reached.")
                    
                    # 1. Run Genetic Algorithm (Optimize Config)
                    evolved = GeneticEvolution.run_evolution_cycle()
                    
                    # 2. Run Neural Training (Update Weights)
                    # We train on top assets to keep the Brain fresh
                    assets_to_train = ["BTC/USDT", "ETH/USDT", "SOL/USDT"] 
                    training_results = self.trainer.run_training_session(assets_to_train, epochs=10)
                    
                    self.last_evolution_time = current_time
                    logger.info(f"🧬 Evolution Cycle Complete. Evolved: {evolved}")
                    
                    self._broadcast_update("SYSTEM_EVENT", {"msg": "System Evolved", "details": f"New Gen: {evolved}"})
                    self._broadcast_wallet_state()
                
                # --- PHASE C: IDLE (Digestion) ---
                self.status = "IDLE"
                # Faster cycle for dynamic news updates
                time.sleep(5) 

            except Exception as e:
                logger.error(f"❌ CRITICAL ERROR in Life Cycle: {e}")
                self.status = "ERROR"
                time.sleep(30) # Prevent tight error loops

    def get_context(self):
        """Returns the current state for the Frontend (UI)"""
        return {
            "status": self.status,
            "scan_results": self.latest_scan_results,
            "last_evolution": datetime.fromtimestamp(self.last_evolution_time).isoformat() if self.last_evolution_time > 0 else "N/A",
            "ai_state": self.ai_core.get_state()
        }

if __name__ == "__main__":
    # Manual Test
    logging.basicConfig(level=logging.INFO)
    node = RedlineAutonomousNode()
    node.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.stop()
