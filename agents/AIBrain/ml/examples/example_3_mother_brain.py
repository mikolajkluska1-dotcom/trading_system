"""
Przykład 3: ZAAWANSOWANY - Mother Brain orkiestruje wszystko
Mother Brain zarządza agentami i uczy się z ich raportów
"""
from agents.AIBrain.ml.mother_brain import MotherBrain
import asyncio

class AutoTradingSystem:
    def __init__(self):
        # Mother Brain zarządza wszystkimi dziećmi
        self.mother = MotherBrain()
        
        # Urodzi 7 dzieci (jeśli jeszcze nie ma)
        if len(self.mother.children) == 0:
            self.mother.birth_child('technical_analyst')
            self.mother.birth_child('volume_hunter')
            self.mother.birth_child('whale_watcher')
            self.mother.birth_child('sentiment_scout')
            self.mother.birth_child('market_scanner')
            self.mother.birth_child('rugpull_detector')
            self.mother.birth_child('report_coordinator')
    
    async def run_trading_loop(self):
        """Główna pętla tradingowa"""
        
        while True:
            try:
                # 1. Mother Brain zbiera raporty od dzieci
                print("\n🧠 Mother Brain: Collecting reports from children...")
                reports = self.mother.collect_reports()
                
                # 2. Mother Brain podejmuje decyzję (używa RL policy)
                print("🧠 Mother Brain: Making decision...")
                decision = self.mother.make_decision(reports)
                
                # 3. Wykonaj trade
                if decision['action'] in ['BUY', 'SELL']:
                    print(f"💰 Executing {decision['action']} on {decision['symbol']}")
                    result = await self.execute_trade(decision)
                    
                    # 4. Mother Brain uczy się z wyniku
                    reward = result['profit']  # Dodatni = zysk, ujemny = strata
                    self.mother.learn_from_trade(decision, reward)
                    
                    # 5. Aktualizuj performance dzieci
                    for child_id, report in reports.items():
                        if report['signal'] == decision['action']:
                            # Dziecko miało rację - nagroda
                            self.mother.update_child_performance(child_id, reward)
                        else:
                            # Dziecko się myliło - kara
                            self.mother.update_child_performance(child_id, -reward)
                
                # 6. Co jakiś czas: ewolucja dzieci
                if self.mother.total_trades % 100 == 0:
                    print("🧬 Mother Brain: Evolving children...")
                    self.mother.evolve_children()
                
                # 7. Zapisz stan
                self.mother.save_state()
                
                # 8. Czekaj na następną iterację
                await asyncio.sleep(3600)  # Co godzinę
                
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(60)
    
    async def execute_trade(self, decision):
        """Wykonaj trade na exchange"""
        import ccxt
        
        exchange = ccxt.binance({
            'apiKey': 'YOUR_API_KEY',
            'secret': 'YOUR_SECRET'
        })
        
        symbol = decision['symbol']
        action = decision['action']
        amount = decision['amount']
        
        if action == 'BUY':
            order = exchange.create_market_buy_order(symbol, amount)
        else:
            order = exchange.create_market_sell_order(symbol, amount)
        
        # Czekaj na wypełnienie i oblicz profit
        # ... (uproszczone)
        
        return {
            'order_id': order['id'],
            'profit': 0  # Oblicz później
        }

# UŻYCIE
if __name__ == "__main__":
    # Uruchom system
    system = AutoTradingSystem()
    
    # Tryb 1: Automatyczny trading (nieskończona pętla)
    # asyncio.run(system.run_trading_loop())
    
    # Tryb 2: Pojedyncza analiza
    reports = system.mother.collect_reports()
    decision = system.mother.make_decision(reports)
    
    print("\n" + "="*60)
    print("MOTHER BRAIN DECISION")
    print("="*60)
    print(f"Action: {decision['action']}")
    print(f"Symbol: {decision['symbol']}")
    print(f"Confidence: {decision['confidence']:.1f}%")
    print(f"\nChild Reports:")
    for child_id, report in reports.items():
        print(f"  {child_id}: {report['signal']} ({report['confidence']:.0f}%)")
    print("="*60)
