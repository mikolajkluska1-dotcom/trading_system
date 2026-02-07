"""
AIBrain v3.0 - Whale Watcher Module (Legacy Compatibility)
=========================================================
Now wraps the v3 child_agent for backward compatibility.
"""
import json
import os
import time
import logging
import requests
from datetime import datetime

logger = logging.getLogger("WHALE_WATCHER")


class WhaleWatcher:
    """
    Whale Watcher & Copy Trading Module - v3 Compatible
    Monitors whale wallets and provides volume spike signals.
    """
    
    # Tier-1 assets where whales don't move markets much
    TIER_1 = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']
    
    def __init__(self, data_path="data/whales.json"):
        self.data_path = data_path
        self.whales = self._load_data()
        self.mock_signals = {} 
        self.processed_txs = set()
        self.api_keys = {
            "ETH": os.getenv("ETHERSCAN_API_KEY", ""),
            "BSC": os.getenv("BSCSCAN_API_KEY", "")
        }
        self.base_urls = {
            "ETH": "https://api.etherscan.io/api",
            "BSC": "https://api.bscscan.com/api"
        }
        
        # v3 DNA
        self.dna = {
            'sensitivity': 0.5,
            'volume_spike_multiplier': 2.0,
            'base_signal': 0.1,  # NEW: Base signal to stay visible
            'whale_threshold': 1000000  # $1M
        }
        
        # v3 state
        self.last_analysis = {'signal': 'HOLD', 'score': 0.1}

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
            "last_block": 0,
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
        """Check for whale signals - v3 enhanced"""
        # Check Mock/Injected first
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

    # ========== v3 Methods ==========
    
    async def analyze(self, market_data):
        """
        v3 Compatible analyze method for Mother Brain
        Returns signal for attention mechanism
        """
        symbol = market_data.get('symbol', '')
        df = market_data.get('df')
        
        base_signal = self.dna.get('base_signal', 0.1)
        
        # Default: base signal (always active)
        if df is None or len(df) < 20:
            self.last_analysis = {
                'signal': 'HOLD', 
                'score': base_signal,
                'reasoning': 'No data - base signal'
            }
            return self.last_analysis
        
        # Volume spike detection
        if 'volume' in df.columns:
            current_vol = df['volume'].iloc[-1]
            avg_vol = df['volume'].rolling(20).mean().iloc[-1]
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                
                if vol_ratio > 2.0:  # Volume spike!
                    price_dir = 1 if df['close'].iloc[-1] > df['close'].iloc[-2] else -1
                    score = price_dir * min(0.8, (vol_ratio - 1) * 0.3)
                    
                    self.last_analysis = {
                        'signal': 'BUY' if score > 0 else 'SELL',
                        'score': score,
                        'reasoning': f'Volume spike {vol_ratio:.1f}x'
                    }
                    return self.last_analysis
        
        # Default: base signal
        self.last_analysis = {
            'signal': 'HOLD', 
            'score': base_signal,
            'reasoning': 'Normal volume'
        }
        return self.last_analysis
    
    def get_signal_for_attention(self) -> float:
        """v3: Return score for Attention mechanism"""
        return self.last_analysis.get('score', 0.1)

    def scan_chain_activity(self):
        """Polls blockchain APIs for new transactions."""
        updates = False
        for whale in self.whales:
            net = whale.get('network', 'ETH')
            addr = whale['address']
            last_blk = whale.get('last_block', 0)
            
            api_key = self.api_keys.get(net)
            if not api_key:
                continue

            url = self.base_urls.get(net)
            if not url:
                continue

            try:
                params = {
                    "module": "account",
                    "action": "txlist",
                    "address": addr,
                    "startblock": last_blk + 1,
                    "endblock": 99999999,
                    "sort": "desc",
                    "apikey": api_key
                }
                
                r = requests.get(url, params=params, timeout=10)
                data = r.json()
                
                if data.get('status') == '1' and data.get('result'):
                    txs = data['result']
                    if txs:
                        # Update last block
                        whale['last_block'] = int(txs[0].get('blockNumber', last_blk))
                        updates = True
                        logger.info(f"🐋 {whale['label']}: {len(txs)} new transactions")

            except Exception as e:
                logger.error(f"Scan failed for {addr}: {e}")

        if updates:
            self._save_data()
