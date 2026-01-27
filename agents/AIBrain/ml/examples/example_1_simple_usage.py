"""
Przykład 1: PROSTY - Bezpośrednie użycie agenta
Używasz jednego agenta do analizy
"""
import torch
from agents.AIBrain.ml.child_agents.base_agent import TechnicalAnalyst

# 1. Załaduj wytrenowanego agenta
agent = TechnicalAnalyst(
    agent_id="technical_analyst_001",
    generation=1
)
agent.load_checkpoint("R:/Redline_Data/ai_models/technical_analyst/technical_analyst_best.pth")

# 2. Przygotuj dane (ostatnie 60 świeczek)
market_data = {
    'symbol': 'BTC/USDT',
    'candles': [
        {'open': 45000, 'high': 45500, 'low': 44800, 'close': 45200, 'volume': 1000},
        {'open': 45200, 'high': 45800, 'low': 45100, 'close': 45600, 'volume': 1200},
        # ... 58 more candles
    ]
}

# 3. Poproś o raport
report = agent.generate_report(market_data)

# 4. Zobacz wynik
print(f"Signal: {report['signal']}")        # BUY, SELL, HOLD
print(f"Confidence: {report['confidence']}")  # 0-100%
print(f"Reason: {report['reason']}")        # "RSI oversold, bullish divergence"

# 5. Podejmij decyzję
if report['signal'] == 'BUY' and report['confidence'] > 70:
    print("🟢 KUPUJ!")
elif report['signal'] == 'SELL' and report['confidence'] > 70:
    print("🔴 SPRZEDAWAJ!")
else:
    print("⚪ CZEKAJ!")
