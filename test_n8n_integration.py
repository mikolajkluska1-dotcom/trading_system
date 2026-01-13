
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from backend.ai_config_manager import AIConfigManager
from backend.ai_core import RedlineAICore
import pandas as pd

def test_integration():
    print("--- 1. Testing Config Persistence ---")
    custom_conf = AIConfigManager.load_config()
    custom_conf['sentiment_weight'] = 80.0 # High impact
    AIConfigManager.save_config(custom_conf)
    
    loaded = AIConfigManager.load_config()
    if loaded['sentiment_weight'] == 80.0:
        print("✅ Config saved and loaded successfully.")
    else:
        print("❌ Config persistence failed.")

    print("\n--- 2. Testing n8n Data Injection ---")
    ai = RedlineAICore(mode="TEST")
    
    # Simulate n8n webhook
    payload = {"sentiment": 90.0, "summary": "Super Bullish News"}
    ai.update_external_context(payload)
    
    if ai.external_context['sentiment'] == 90.0:
        print("✅ External context updated.")
    else:
        print("❌ Context update failed.")

    print("\n--- 3. Testing Logic Impact ---")
    # Mock Market Data
    market_data = {"close": 100, "rsi": 50, "volatility": 0.01}
    df = pd.DataFrame({'close': [100]*60}) # Mock DF for ML
    
    # Evaluate
    # Sentiment 90 (Neutral is 50) -> +40 delta. Weight 80% -> +32 Score impact!
    decision = ai.evaluate("BTC/USDT", market_data, df, loaded)
    
    print(f"Decision Score: {decision.score}")
    print(f"Reasons: {decision.reasons}")
    
    if any("Sentiment" in r for r in decision.reasons):
        print("✅ n8n Sentiment impacted the decision logic!")
    else:
        print("❌ Sentiment ignored in logic.")
        
if __name__ == "__main__":
    test_integration()
