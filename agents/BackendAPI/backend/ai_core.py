# backend/ai_core.py
"""
REDLINE AI CORE — GEN 3.8 (HYBRID INTELLIGENCE)
Integration: Logic Engine + Neural Network (DeepBrain) + External Agentic Data (n8n)
"""

import logging
import hashlib
from datetime import datetime

# ML Imports (zakładamy, że te pliki istnieją, jeśli nie - system może zgłosić błąd importu)
# Jeśli nie używasz jeszcze ML, te importy mogą wymagać "zaślepienia"
try:
    from agents.AIBrain.ml.brain import DeepBrain
    from agents.AIBrain.ml.whale_watcher import WhaleWatcher
except ImportError:
    # Fallback dla trybu testowego bez pełnego ML
    class DeepBrain:
        def predict(self, df): return 0, 0, "NEUTRAL"
    class WhaleWatcher:
        def check_for_signals(self, s): return None

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

        # --- SHORT TERM MEMORY (n8n Integration) ---
        self.market_sentiment = 0.0
        self.whale_pressure = 0.0

        # Dwa półkule mózgu
        self.logic_brain = DecisionEngine()
        self.neural_brain = DeepBrain() # ML V7
        self.whale_watcher = WhaleWatcher() # Copy Trading Module

        # --- ORCHESTRATOR OWNERSHIP (Unified Architecture) ---
        # Local import to prevent circular dependency
        try:
            from agents.Database.core.orchestrator import Orchestrator
            self.orchestrator = Orchestrator(mode=mode, ai_core=self)
        except ImportError:
            logger.warning("Orchestrator not found, running in standalone mode.")
            class MockOrch: 
                is_running=False
                def set_autopilot(self, x): pass
            self.orchestrator = MockOrch()

        logger.info(f"AI CORE INITIALIZED: {self.state}")

    def start(self):
        """Activates the Orchestrator loop."""
        if not self.orchestrator.is_running:
            self.orchestrator.is_running = True
            self.orchestrator.set_autopilot(True) 
            logger.info("🟢 AI SYSTEM STARTED (Orchestrator Online)")

    def stop(self):
        """Stops the Orchestrator loop."""
        if self.orchestrator.is_running:
            self.orchestrator.is_running = False
            self.orchestrator.set_autopilot(False)
            logger.info("🔴 AI SYSTEM STOPPED (Orchestrator Offline)")

    def get_state(self):
        full_state = self.state.copy()
        full_state["external_context"] = self.external_context
        full_state["short_term_memory"] = {
            "market_sentiment": self.market_sentiment,
            "whale_pressure": self.whale_pressure
        }
        return full_state

    def process_webhook_data(self, data: dict):
        """
        Updates Short Term Memory based on n8n webhooks.
        Fixed to be robust against None/Null values.
        """
        try:
            source = data.get("source", "unknown")
            logger.info(f"📦 RAW PACKET RECEIVED: {data}")

            # Wyciągamy payload (czasem jest w 'data', czasem płasko)
            payload = data.get("data", data)
            if not payload: 
                payload = data # Fallback

            # 1. News / Sentiment
            if source in ["news", "sentiment", "n8n-news-agent"]:
                # Szukamy wartości pod różnymi kluczami
                val = payload.get("sentiment")
                if val is None: val = payload.get("value")
                if val is None: val = payload.get("sentiment_score")

                if val is not None:
                    try:
                        self.market_sentiment = float(val)
                    except (ValueError, TypeError):
                        pass # Ignorujemy błędy parsowania

            # 2. Whale Watcher
            elif source == "whale_watcher":
                msg_type = payload.get("type", "UNKNOWN")
                
                # Bezpieczne pobranie wartości (zamiana None na 0)
                raw_val = payload.get("value")
                if raw_val is None: raw_val = 0
                
                try:
                    usd_value = float(raw_val)
                except (ValueError, TypeError):
                    usd_value = 0.0
                
                # Impact Logic: Every $100k = 1 point
                impact = usd_value / 100000.0
                
                if "BUY" in msg_type or "GREEN" in msg_type:
                    self.whale_pressure += impact
                elif "SELL" in msg_type or "RED" in msg_type:
                    self.whale_pressure -= impact
                
                # Clamp between -100 and 100
                self.whale_pressure = max(-100.0, min(100.0, self.whale_pressure))

            logger.info(f"🧠 MEMORY UPDATED | Sentiment: {self.market_sentiment:.1f} | Whale Pressure: {self.whale_pressure:.1f}")

        except Exception as e:
            logger.error(f"⚠️ Webhook Processing Error: {e}")

    def update_external_context(self, data):
        """Legacy wrapper"""
        if "sentiment" in data:
            data["source"] = "sentiment"
            self.process_webhook_data(data)
            self.external_context["sentiment"] = float(data["sentiment"])
        
        if "summary" in data:
            self.external_context["summary"] = data["summary"]
            
        self.external_context["last_update"] = datetime.utcnow().isoformat()

    def set_mode(self, mode):
        self.state["mode"] = mode.upper()

    def _generate_id(self, symbol, score, action):
        raw = f"{symbol}|{score}|{action}|{datetime.utcnow().timestamp()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def evaluate(self, symbol, market_data, df, config):
        """
        Główna metoda decyzyjna.
        """
        # 1. Logika
        l_score, l_conf, l_reasons, l_flags = self.logic_brain.analyze(symbol, market_data)

        # 2. ML (Zabezpieczenie przed brakiem danych)
        if df is not None and not df.empty:
            ml_pred, ml_conf, ml_sig = self.neural_brain.predict(df)
        else:
            ml_pred, ml_conf, ml_sig = 0, 0, "NEUTRAL"

        # 3. Fuzja Danych
        final_score = l_score 
        
        # Add Memory Factors
        final_score += self.market_sentiment
        final_score += self.whale_pressure
        
        if abs(self.market_sentiment) > 1:
            l_reasons.append(f"Sentiment Impact ({self.market_sentiment:+.1f})")
        
        if abs(self.whale_pressure) > 1:
            l_reasons.append(f"Whale Pressure ({self.whale_pressure:+.1f})")
            
        final_conf = l_conf
        
        # 4. Copy Trading / ML Impact (Uproszczone)
        if ml_conf > 0.6:
            if ml_sig == "BUY":
                final_score += 15
                l_reasons.append(f"ML Confirms BUY ({ml_conf:.2f})")
            elif ml_sig == "SELL":
                final_score -= 15
                l_reasons.append(f"ML Confirms SELL ({ml_conf:.2f})")

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
            logger.info(f"🧠 HYBRID SIGNAL: {symbol} {action} (Score: {final_score:.0f})")

        return TradeSignal(
            symbol,
            round(final_score, 1),
            round(final_conf, 2),
            action,
            l_reasons,
            l_flags,
            decision_id,
            ml_data={"sig": ml_sig, "whale_pressure": self.whale_pressure}
        )