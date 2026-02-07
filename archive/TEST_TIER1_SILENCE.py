"""
AIBrain v3.0 - Agent Diagnostics
Tests Tier-1 Silence logic for Rugpull Detector
"""
import asyncio
import sys
sys.path.insert(0, "c:/Users/Mikołaj/trading_system")

from agents.AIBrain.ml.child_agents.rugpull_detector_agent import RugpullDetectorAgent
from agents.AIBrain.ml.child_agents.scanner_agent import ScannerAgent
from agents.AIBrain.ml.child_agents.technician_agent import TechnicianAgent

print("=" * 60)
print("🧪 AGENT DIAGNOSTICS - TIER-1 SILENCE TEST")
print("=" * 60)

async def test_agents():
    # Initialize agents
    rugpull = RugpullDetectorAgent()
    scanner = ScannerAgent()
    technician = TechnicianAgent()
    
    print(f"\n🔬 Testing RugpullDetector DNA:")
    print(f"   tier1_silence: {rugpull.dna.get('tier1_silence')}")
    print(f"   SAFE_ASSETS: {rugpull.SAFE_ASSETS}")
    
    # Test symbols
    test_cases = [
        ('BTCUSDT', 'Tier-1 Safe'),
        ('ETHUSDT', 'Tier-1 Safe'),
        ('SOLUSDT', 'Tier-1 Safe'),
        ('BNBUSDT', 'Tier-1 Safe'),
        ('SHITCOIN', 'Unknown (should be neutral)'),
        ('PEPEUSDT', 'Meme coin'),
    ]
    
    print(f"\n📊 RUGPULL DETECTOR RESPONSES:")
    print("-" * 60)
    
    for symbol, desc in test_cases:
        result = await rugpull.analyze({'symbol': symbol, 'df': None})
        score = result.get('score', 0)
        signal = result.get('signal', 'N/A')
        reasoning = result.get('reasoning', 'N/A')[:40]
        
        # Expected: Tier-1 should be 0.0
        status = "✅" if (symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'] and score == 0.0) else "⚠️"
        
        print(f"   {status} {symbol:12} | Signal: {signal:8} | Score: {score:5.2f} | {reasoning}")
    
    print("-" * 60)
    
    # Summary
    print(f"\n🎯 EXPECTED BEHAVIOR:")
    print(f"   - Tier-1 (BTC, ETH, SOL, BNB): score = 0.0 (invisible to Mother Brain)")
    print(f"   - Other coins: score varies based on risk analysis")
    
    print(f"\n✅ Diagnostics complete!")

if __name__ == "__main__":
    asyncio.run(test_agents())
