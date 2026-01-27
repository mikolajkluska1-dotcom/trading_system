import sys
import os
import time

# Force CPU to avoid CUDA hangs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import logging

# Setup path
sys.path.append(os.getcwd())

from agents.BackendAPI.backend.ai_core import RedlineAICore

# Config for test
TEST_CONFIG = {
    "copy_trading_enabled": True,
    "whale_trust_factor": 1.0, 
    "min_confidence": 0.5,
    "sentiment_weight": 50
}

def verify_whale_integration():
    print("🐋 STARTING WHALE WATCHER VERIFICATION 🐋", flush=True)
    
    try:
        # 1. Initialize Core
        core = RedlineAICore(mode="PAPER")
        print("Core Initialized.", flush=True)
        
        # 2. Add a Whale
        core.whale_watcher.add_whale("0x123...abc", "Elon Musk Mock", "ETH")
        whales = core.whale_watcher.get_whales()
        print(f"Whales Tracked: {len(whales)}", flush=True)
        
        # 3. Simulate Market Data (Neutral)
        symbol = "DOGE/USDT"
        market_ctx = {
            "close": 0.15,
            "rsi": 50.0, 
            "volatility": 0.01,
            "sma_50": 0.15
        }
        df = pd.DataFrame([{
            "open": 0.15, "high": 0.15, "low": 0.15, "close": 0.15, "volume": 1000
        }])
        
        # 4. Baseline
        decision_base = core.evaluate(symbol, market_ctx, df, TEST_CONFIG)
        print(f"Base Score: {decision_base.score}", flush=True)
        
        # 5. Inject Whale Buy
        print("Injecting Signal...", flush=True)
        core.whale_watcher.inject_signal(symbol, "BUY", "0x123...abc")
        
        # 6. Evaluate
        decision_whale = core.evaluate(symbol, market_ctx, df, TEST_CONFIG)
        print(f"Whale Score: {decision_whale.score}", flush=True)
        print(f"Reasons: {decision_whale.reasons}", flush=True)
        
        if decision_whale.score > decision_base.score:
            print("✅ SUKCES: Whale Signal boosted the score!", flush=True)
        else:
            print("❌ FAILURE: No score boost.", flush=True)
            
    except Exception as e:
        print(f"❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_whale_integration()
