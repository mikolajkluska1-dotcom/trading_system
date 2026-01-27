import json
import os
import time
import logging
import requests
from datetime import datetime

logger = logging.getLogger("WHALE_WATCHER")

class WhaleWatcher:
    """
    Whale Watcher & Copy Trading Module.
    Monitors a list of 'Whale' wallets.
    """
    
    def __init__(self, data_path="data/whales.json"):
        self.data_path = data_path
        self.whales = self._load_data()
        self.mock_signals = {} 
        self.processed_txs = set() # Cache to avoid duplicate signals
        self.api_keys = {
            "ETH": "YOUR_ETHERSCAN_KEY", # TODO: Move to config
            "BSC": "YOUR_BSCSCAN_KEY"
        }
        self.base_urls = {
            "ETH": "https://api.etherscan.io/api",
            "BSC": "https://api.bscscan.com/api"
        }

    def _load_data(self):
        if not os.path.exists(self.data_path):
            return []
        try:
            with open(self.data_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_data(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(self.whales, f, indent=4)

    def add_whale(self, address, label, network="ETH"):
        for w in self.whales:
            if w['address'] == address:
                return False, "Already exists"
        
        self.whales.append({
            "address": address,
            "label": label,
            "network": network,
            "trust_score": 80.0,
            "last_block": 0, # Tracker
            "added_at": datetime.utcnow().isoformat()
        })
        self._save_data()
        msg = f"🐋 WHALE ADDED: {label} ({address[:6]}...)"
        logger.info(msg)
        return True, msg

    def remove_whale(self, address):
        initial_len = len(self.whales)
        self.whales = [w for w in self.whales if w['address'] != address]
        if len(self.whales) < initial_len:
            self._save_data()
            return True
        return False

    def get_whales(self):
        return self.whales

    def check_for_signals(self, symbol):
        # 1. Check Mock/Injected
        if symbol in self.mock_signals:
            sig = self.mock_signals[symbol]
            if time.time() - sig['ts'] < 3600:
                return sig
        return None

    def inject_signal(self, symbol, side, whale_address="0xMockWhale"):
        whale_label = "Unknown"
        whale_trust = 50.0
        for w in self.whales:
            if w['address'] == whale_address:
                whale_label = w['label']
                whale_trust = w['trust_score']
                break

        self.mock_signals[symbol] = {
            "whale": whale_label,
            "address": whale_address,
            "side": side, 
            "trust_score": whale_trust,
            "ts": time.time()
        }
        logger.info(f"🐋 WHALE SIGNAL INJECTED: {whale_label} -> {side} {symbol}")

    def scan_chain_activity(self):
        """
        Polls blockchain APIs for new transactions.
        """
        updates = False
        for whale in self.whales:
            net = whale.get('network', 'ETH')
            addr = whale['address']
            last_blk = whale.get('last_block', 0)
            
            # Simple simulation if no key (Fail-safe)
            if self.api_keys.get(net) == "YOUR_ETHERSCAN_KEY":
                continue # Skip real fetch if no key

            url = self.base_urls.get(net)
            if not url: continue

            try:
                # Etherscan Standard API: Get Normal Tx
                params = {
                    "module": "account",
                    "action": "txlist",
                    "address": addr,
                    "startblock": last_blk + 1,
                    "endblock": 99999999,
                    "sort": "desc",
                    "apikey": self.api_keys.get(net)
                }
                
                # Fetch
                # r = requests.get(url, params=params, timeout=5)
                # data = r.json()
                
                # Implementation Note:
                # Real implementation requires parsing 'input' data to see if it's a Swap.
                # For this step, we keep the structure ready for the user to key-in.
                pass 

            except Exception as e:
                logger.error(f"Scan failed for {addr}: {e}")

        if updates:
            self._save_data()
