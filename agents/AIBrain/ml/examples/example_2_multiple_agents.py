"""
Przykład 2: ŚREDNI - Użycie wielu agentów razem
Każdy agent analizuje swój aspekt, ty łączysz wyniki
"""
from agents.AIBrain.ml.child_agents.base_agent import (
    TechnicalAnalyst,
    VolumeHunter,
    WhaleWatcher
)

class TradingDecisionMaker:
    def __init__(self):
        # Załaduj wszystkich agentów
        self.technical = TechnicalAnalyst(agent_id="tech_001", generation=1)
        self.volume = VolumeHunter(agent_id="vol_001", generation=1)
        self.whale = WhaleWatcher(agent_id="whale_001", generation=1)
        
        # Załaduj ich modele
        self.technical.load_checkpoint("R:/Redline_Data/ai_models/technical_analyst/best.pth")
        self.volume.load_checkpoint("R:/Redline_Data/ai_models/volume_hunter/best.pth")
        self.whale.load_checkpoint("R:/Redline_Data/ai_models/whale_watcher/best.pth")
    
    def analyze(self, symbol):
        """Zbierz raporty od wszystkich agentów"""
        
        # 1. Pobierz dane
        market_data = self.get_market_data(symbol)
        onchain_data = self.get_onchain_data(symbol)
        
        # 2. Każdy agent robi swoją robotę
        tech_report = self.technical.generate_report(market_data)
        vol_report = self.volume.generate_report(market_data)
        whale_report = self.whale.generate_report(onchain_data)
        
        # 3. Połącz wyniki
        return self.combine_reports(tech_report, vol_report, whale_report)
    
    def combine_reports(self, tech, vol, whale):
        """Prosta logika łączenia"""
        
        # Zlicz głosy
        buy_votes = 0
        sell_votes = 0
        
        if tech['signal'] == 'BUY': buy_votes += tech['confidence']
        if tech['signal'] == 'SELL': sell_votes += tech['confidence']
        
        if vol['signal'] == 'BUY': buy_votes += vol['confidence']
        if vol['signal'] == 'SELL': sell_votes += vol['confidence']
        
        if whale['signal'] == 'BUY': buy_votes += whale['confidence']
        if whale['signal'] == 'SELL': sell_votes += whale['confidence']
        
        # Decyzja
        if buy_votes > sell_votes and buy_votes > 150:
            return {
                'action': 'BUY',
                'confidence': buy_votes / 3,
                'reasons': [tech['reason'], vol['reason'], whale['reason']]
            }
        elif sell_votes > buy_votes and sell_votes > 150:
            return {
                'action': 'SELL',
                'confidence': sell_votes / 3,
                'reasons': [tech['reason'], vol['reason'], whale['reason']]
            }
        else:
            return {'action': 'HOLD', 'confidence': 50, 'reasons': ['Mixed signals']}
    
    def get_market_data(self, symbol):
        # Pobierz z exchange
        import ccxt
        exchange = ccxt.binance()
        candles = exchange.fetch_ohlcv(symbol, '1h', limit=60)
        return {'symbol': symbol, 'candles': candles}
    
    def get_onchain_data(self, symbol):
        # Pobierz z blockchain
        return {'symbol': symbol, 'whale_movements': []}

# UŻYCIE
if __name__ == "__main__":
    trader = TradingDecisionMaker()
    
    # Analizuj BTC
    decision = trader.analyze('BTC/USDT')
    
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.1f}%")
    print(f"Reasons:")
    for reason in decision['reasons']:
        print(f"  - {reason}")
