# backend/ai_core.py
"""
REDLINE AI CORE — GEN 3.8 (HYBRID INTELLIGENCE)
Integration: Logic Engine + Neural Network (DeepBrain)
"""

import logging
import hashlib
from datetime import datetime

# ML Imports
from ml.brain import DeepBrain

# Logging setup
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("AI_CORE")

class TradeSignal:
    """Struktura wyniku decyzji"""
    def __init__(self, symbol, score, confidence, action, reasons, flags, decision_id, ml_data=None):
        self.symbol = symbol
        self.score = score           # 0-100
        self.confidence = confidence # 0.0 - 1.0
        self.action = action         # STRONG_BUY, BUY, HOLD, SELL...
        self.reasons = reasons       # Lista uzasadnień
        self.flags = flags           # Flagi ryzyka
        self.decision_id = decision_id
        self.ml_data = ml_data or {} # Wyniki z sieci neuronowej
        self.timestamp = datetime.utcnow().isoformat()

    def __repr__(self):
        return f"Signal({self.symbol}: {self.action} | Score: {self.score} | Conf: {self.confidence:.2f})"

class DecisionEngine:
    """Silnik Logiczny (Regułowy)"""
    def __init__(self):
        self.weights = {"RSI": 25, "TREND": 25, "VOLATILITY_PENALTY": 20}

    def analyze(self, symbol, market_data):
        """Zwraca bazowy Score na podstawie analizy technicznej"""
        score = 50.0
        confidence = 1.0
        reasons = []
        flags = []

        price = market_data.get('close', 0.0)
        if price <= 0: return 50, 0, "HOLD", ["Invalid Data"], ["ERR"]

        # 1. Volatility Penalty
        volatility = market_data.get('volatility', 0.0)
        if volatility > 0.03:
            flags.append("HIGH_VOLATILITY")
            confidence -= 0.3
            score -= 10
            reasons.append(f"High Volatility ({volatility:.1%})")

        # 2. RSI
        rsi = market_data.get('rsi', 50.0)
        if rsi < 30:
            score += 20
            reasons.append(f"RSI Oversold ({rsi:.0f})")
        elif rsi > 70:
            score -= 20
            reasons.append(f"RSI Overbought ({rsi:.0f})")

        # 3. Trend
        sma = market_data.get('sma_50', price)
        if price > sma * 1.005:
            score += 15
            reasons.append("Uptrend (Price > SMA)")
        elif price < sma * 0.995:
            score -= 15
            reasons.append("Downtrend (Price < SMA)")

        return score, confidence, reasons, flags

class RedlineAICore:
    """
    HYBRID ORCHESTRATOR
    Łączy logikę klasyczną (DecisionEngine) z intuicją ML (DeepBrain).
    """
  
    def __init__(self, mode="PAPER", timeframe="1h"):
  
        self.state = {
            "mode": mode,
            "running": True,
            "timeframe": timeframe,
            "engine": "GEN-3.8 (Hybrid)"
        }
        # Dwa półkule mózgu
        self.logic_brain = DecisionEngine()
        self.neural_brain = DeepBrain() # ML V7
        
        logger.info(f"AI CORE INITIALIZED: {self.state}")

    def get_state(self):
        return self.state

    def set_mode(self, mode):
        self.state["mode"] = mode.upper()

    def _generate_id(self, symbol, score, action):
        raw = f"{symbol}|{score}|{action}|{datetime.utcnow().timestamp()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def evaluate(self, symbol, market_data, df, config):
        """
        Główna metoda decyzyjna.
        Wymaga: market_data (dict) ORAZ df (DataFrame dla ML)
        """
        # 1. Lewa półkula: Logika (Reguły)
        l_score, l_conf, l_reasons, l_flags = self.logic_brain.analyze(symbol, market_data)

        # 2. Prawa półkula: Intuicja (DeepBrain ML)
        # Zwraca: (price_pred, ml_conf, ml_signal)
        ml_pred, ml_conf, ml_sig = self.neural_brain.predict(df)
        
        # 3. Fuzja Danych (Hybrid Consensus)
        final_score = l_score
        final_conf = l_conf
        
        # ML Impact - Jeśli ML jest pewne (>0.6), wpływa na wynik
        if ml_conf > 0.6:
            if ml_sig == "BUY":
                final_score += 15
                l_reasons.append(f"ML Confirms BUY (Conf {ml_conf:.2f})")
            elif ml_sig == "SELL":
                final_score -= 15
                l_reasons.append(f"ML Confirms SELL (Conf {ml_conf:.2f})")
            
            # Boost pewności jeśli logika i ML się zgadzają
            if (l_score > 60 and ml_sig == "BUY") or (l_score < 40 and ml_sig == "SELL"):
                final_conf = min(1.0, final_conf + 0.15)
                l_reasons.append("HYBRID CONFLUENCE") # Najsilniejszy sygnał
        
        # OOD Protection (z DeepBraina)
        if "OOD" in ml_sig:
            final_conf = 0.0
            l_reasons.append(f"ML Veto: {ml_sig}")
            l_flags.append("AI_OOD_EVENT")

        # 4. Finalna Klasyfikacja
        final_score = max(0, min(100, final_score))
        min_conf = config.get("min_confidence", 0.6)
        
        action = "HOLD"
        if final_conf >= min_conf:
            if final_score >= 85: action = "STRONG_BUY"
            elif final_score >= 65: action = "BUY"
            elif final_score <= 15: action = "STRONG_SELL"
            elif final_score <= 35: action = "SELL"
        else:
            l_reasons.append(f"Low Confidence ({final_conf:.2f})")

        # 5. Wynik
        decision_id = self._generate_id(symbol, final_score, action)
        
        # Logujemy tylko silne sygnały, żeby nie śmiecić
        if "STRONG" in action:
            logger.info(f"🧠 HYBRID SIGNAL: {symbol} {action} (Score: {final_score:.0f}, ML Conf: {ml_conf:.2f})")

        return TradeSignal(
            symbol, 
            round(final_score, 1), 
            round(final_conf, 2), 
            action, 
            l_reasons, 
            l_flags, 
            decision_id,
            ml_data={"pred": ml_pred, "conf": ml_conf, "sig": ml_sig}
        )