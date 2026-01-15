import time
import requests
import threading
import logging
import json
from datetime import datetime

# Component Imports
from backend.ai_core import RedlineAICore
from ml.scanner import MarketScanner
from ml.training import OfflineTrainer
from ml.evolution import GeneticEvolution

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

    def _life_cycle(self):
        """The Infinite Loop of the Organism."""
        while self.running:
            try:
                # --- PHASE A: SCANNING (Senses) ---
                self.status = "SCANNING"
                logger.info(f"👁️ Phase A: Scanning Markets... [{datetime.now().strftime('%H:%M:%S')}]")
                
                # Run Scanner
                results = self.scanner.run(self.config)
                if results:
                    self.latest_scan_results = results
                    logger.info(f"👁️ Scan Complete. Found {len(results)} opportunities.")
                    self._broadcast_update("SCAN_COMPLETE", results)
                else:
                    logger.info("👁️ Scan Complete. No opportunities found.")

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
                
                # --- PHASE C: IDLE (Digestion) ---
                self.status = "IDLE"
                # logger.info("💤 Phase C: Resting for 60s...")
                time.sleep(60)

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
