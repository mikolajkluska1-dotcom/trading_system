# backend/ai_core.py
"""
REDLINE AI CORE — GEN 3.8 (HYBRID INTELLIGENCE)
Integration: Logic Engine + Neural Network (DeepBrain) + External Agentic Data (n8n)
"""

import logging
import hashlib
from datetime import datetime

# ML Imports
from ml.brain import DeepBrain
from ml.whale_watcher import WhaleWatcher

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
    Łączy logikę klasyczną (DecisionEngine), intuicję ML (DeepBrain) oraz
    zewnętrzne dane agentowe (External Context / n8n).
    """

    def __init__(self, mode="PAPER", timeframe="1h"):

        self.state = {
            "mode": mode,
            "running": True,
            "timeframe": timeframe,
            "engine": "GEN-3.8 (Hybrid + Agentic)"
        }
        
        # Pamięć krótkotrwała dla danych z zewnątrz (n8n / agenty newsowe)
        self.external_context = {
            "sentiment": 50.0, # 0-100 (50 = Neutral)
            "last_update": None,
            "summary": "No external data yet."
        }

        # Dwa półkule mózgu
        self.logic_brain = DecisionEngine()
        self.neural_brain = DeepBrain() # ML V7
        self.whale_watcher = WhaleWatcher() # Copy Trading Module

        # --- ORCHESTRATOR OWNERSHIP (Unified Architecture) ---
        # Local import to prevent circular dependency:
        # Orchestrator -> MarketScanner -> RedlineAICore
        from core.orchestrator import Orchestrator
        self.orchestrator = Orchestrator(mode=mode, ai_core=self)

        logger.info(f"AI CORE INITIALIZED: {self.state}")

    def start(self):
        """Activates the Orchestrator loop."""
        if not self.orchestrator.is_running:
            self.orchestrator.is_running = True
            # Also set the legacy Orchestrator autopilot if needed
            self.orchestrator.set_autopilot(True) 
            logger.info("🟢 AI SYSTEM STARTED (Orchestrator Online)")

    def stop(self):
        """Stops the Orchestrator loop."""
        if self.orchestrator.is_running:
            self.orchestrator.is_running = False
            self.orchestrator.set_autopilot(False)
            logger.info("🔴 AI SYSTEM STOPPED (Orchestrator Offline)")

    def get_state(self):
        # Dołączamy external context do stanu
        full_state = self.state.copy()
        full_state["external_context"] = self.external_context
        return full_state

    def update_external_context(self, data):
        """
        Metoda dla Webhooka (n8n)
        Oczekuje: {"sentiment": float, "summary": str}
        """
        if "sentiment" in data:
            self.external_context["sentiment"] = float(data["sentiment"])
        if "summary" in data:
            self.external_context["summary"] = data["summary"]
        self.external_context["last_update"] = datetime.utcnow().isoformat()
        logger.info(f"🧠 EXT CONTEXT UPDATED: {self.external_context}")

    def set_mode(self, mode):
        self.state["mode"] = mode.upper()

    def _generate_id(self, symbol, score, action):
        raw = f"{symbol}|{score}|{action}|{datetime.utcnow().timestamp()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def evaluate(self, symbol, market_data, df, config):
        """
        Główna metoda decyzyjna.
        Wymaga: market_data (dict) ORAZ df (DataFrame dla ML) ORAZ config (dict)
        """
        # 1. Lewa półkula: Logika (Reguły)
        l_score, l_conf, l_reasons, l_flags = self.logic_brain.analyze(symbol, market_data)

        # 2. Prawa półkula: Intuicja (DeepBrain ML)
        ml_pred, ml_conf, ml_sig = self.neural_brain.predict(df)

        # 3. Trzecie Oko: External Context (Sentiment / n8n)
        ext_sentiment = self.external_context["sentiment"]
        sent_weight = config.get("sentiment_weight", 50.0) / 100.0 # Normalizacja 0-1
        
        # Wpływ sentymentu na wynik podstawowy
        sentiment_delta = (ext_sentiment - 50.0) * sent_weight 
        # Np. sentiment 80 (bullish), waga 1.0 -> +30 pkt do score
        
        # 4. Fuzja Danych (Hybrid Consensus)
        final_score = l_score + sentiment_delta
        final_conf = l_conf
        
        if abs(sentiment_delta) > 5:
            l_reasons.append(f"Market Sentiment Impact ({sentiment_delta:+.1f})")

        # --- WHALE COPY TRADING INTEGRATION ---
        if config.get("copy_trading_enabled", False):
            whale_sig = self.whale_watcher.check_for_signals(symbol)
            if whale_sig:
                # Calculate Impact
                trust = whale_sig['trust_score'] / 100.0
                factor = config.get("whale_trust_factor", 0.5)
                
                # Signal Matching
                if whale_sig['side'] == "BUY":
                     # Boost Score
                     boost = 25.0 * trust * factor
                     final_score += boost
                     l_reasons.append(f"🐋 WHALE BUY: {whale_sig['whale']} (+{boost:.1f})")
                     
                     # Force Confidence if trust is high
                     if trust > 0.8:
                         final_conf = max(final_conf, 0.75)
                         l_reasons.append("Whale Confidence Boost")

                elif whale_sig['side'] == "SELL":
                     penalty = 25.0 * trust * factor
                     final_score -= penalty
                     l_reasons.append(f"🐋 WHALE SELL: {whale_sig['whale']} (-{penalty:.1f})")

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

        # OOD Protection
        if "OOD" in ml_sig:
            final_conf = 0.0
            l_reasons.append(f"ML Veto: {ml_sig}")
            l_flags.append("AI_OOD_EVENT")

        # 5. Finalna Klasyfikacja
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

        # 6. Wynik
        decision_id = self._generate_id(symbol, final_score, action)

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
