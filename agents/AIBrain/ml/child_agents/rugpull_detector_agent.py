"""
AIBrain v4.0 - HUNTER UPDATE
Rugpull Detector Agent - THE GUARDIAN
Strategia: Tier-1 Silence + Dead Volume Detection
"""
from .base_agent import BaseAgent


class RugpullDetectorAgent(BaseAgent):
    def __init__(self, name="rugpull_detector"):
        super().__init__(name, specialty="scam_detection")
        # Na tych parach ryzyko Rugpulla jest zerowe.
        # Agent ma być dla nich "przezroczysty".
        self.TIER_1_ASSETS = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 
            'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
            'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'ATOMUSDT'
        ]

    def _initialize_dna(self) -> dict:
        return {
            'tier1_silence': True,
            'dead_volume_threshold': 100,
            'holder_concentration_risk': 0.5  # Placeholder for future on-chain
        }

    async def analyze(self, market_data):
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # LOGIKA 1: GŁÓWNE KRYPTOWALUTY (TIER 1) -> IGNORUJ
        # Zwracamy neutralne 0.0, aby Attention Router szukał sygnałów u Skanera/Technika.
        if symbol in self.TIER_1_ASSETS or any(t in symbol.upper() for t in ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'XRP', 'DOGE']):
            self.last_analysis = {
                'signal': 'NEUTRAL', 
                'score': 0.0, 
                'reasoning': "Tier 1 Asset (Safe)"
            }
            return self.last_analysis
        
        # LOGIKA 2: RESZTA RYNKU (Placeholder pod przyszłą logikę on-chain)
        # Tu w przyszłości wepniemy sprawdzanie kontraktów (HoneyPot check).
        # Na razie prosty filtr wolumenu.
        
        df = market_data.get('klines') or market_data.get('df')
        volume = 0
        if df is not None and len(df) > 0:
            volume = df.iloc[-1]['volume'] if 'volume' in df.columns else 0
        else:
            volume = market_data.get('volume', 0)
        
        # Jeśli wolumen jest podejrzanie niski (martwy coin) -> VETO
        if volume < self.dna['dead_volume_threshold']: 
            self.last_analysis = {
                'signal': 'VETO',
                'score': -1.0,
                'reasoning': f"Dead Volume ({volume:.0f}) - Risk High"
            }
            return self.last_analysis

        # Domyślnie neutralnie dla nieznanych coinów
        self.last_analysis = {
            'signal': 'NEUTRAL', 
            'score': 0.0,
            'reasoning': "Unknown asset - neutral"
        }
        return self.last_analysis
    
    def get_signal_for_attention(self) -> float:
        return self.last_analysis.get('score', 0.0)
