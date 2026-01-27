import random
from typing import List, Dict, Optional
import datetime

class AIModel:
    def __init__(self, model_id: str, name: str, description: str, system_prompt: str):
        self.id = model_id
        self.name = name
        self.description = description
        self.system_prompt = system_prompt

    def generate_response(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Mock response generator based on keywords.
        In a real system, this would call an LLM.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M")
        
        # Simple keyword matching for demo purposes
        msg_lower = message.lower()
        
        prefix = ""
        content = ""

        if "hello" in msg_lower or "hi" in msg_lower:
            content = f"Hello. I am {self.name}. How can I assist you with the market today?"
        elif "status" in msg_lower:
            content = "All systems operational. Market data flow is stable."
        elif "help" in msg_lower:
            content = self.description
        else:
            # Fallback to model-specific "thinking"
            content = self._get_specific_response(msg_lower)

        return content

    def _get_specific_response(self, msg: str) -> str:
        return "I've noted that. Let me analyze the implications..."

# --- SPECIALIZED CHILDREN MODELS (SONS) ---

class ScannerSon(AIModel):
    def _get_specific_response(self, msg: str) -> str:
        responses = [
            "Scanning 500+ pairs... Detect abnormal volume on SOL.",
            "Whale alert detected. Tracking large movement.",
            "Market breadth is thinning. Be careful.",
            "I found 3 potential setups matching your criteria.",
            "Scanning completed. No high-probability setups found right now."
        ]
        return random.choice(responses)

class TechnicalSon(AIModel):
    def _get_specific_response(self, msg: str) -> str:
        responses = [
            "RSI is overbought on the 4H timeframe. Watch for pullback.",
            "MACD crossover imminent. Bullish divergence detected.",
            "Support at $142.50 is holding strong.",
            "Price action suggests a breakout pattern forming.",
            "Fibonacci retracement level rejected the price."
        ]
        return random.choice(responses)

class RugpullSon(AIModel): # Formerly Risk Manager
    def _get_specific_response(self, msg: str) -> str:
        responses = [
            "WARNING: Contract ownership is not renounced for this token.",
            "Liquidity is unlocked. High risk of rugpull.",
            "This token has mutable metadata. Exercise extreme caution.",
            "Honeypot check: PASSED. But developer wallet holds 15%.",
            "I advise against this trade. Risk/Reward ratio is poor."
        ]
        return random.choice(responses)

# --- MOTHER MODEL (MAIN) ---

class MotherAI(AIModel):
    def _get_specific_response(self, msg: str) -> str:
        responses = [
            "I am coordinating my sons. Scanner is active, Rugpull is monitoring.",
            "My sons report a mixed market sentiment. Proceed with caution.",
            "I have updated the global strategy based on new data.",
            "System integrity is 100%. Neural lattice is stable.",
            "I am the Mother. I oversee all operations."
        ]
        return random.choice(responses)


# --- INSTANCES ---

models_list = [
    MotherAI(
        model_id="mother",
        name="AI Mother (Main)",
        description="The central orchestrator. Oversees all operations and coordinates other agents.",
        system_prompt="You are the Mother AI, the central brain of the Redline system. You are calm, authoritative, and omniscient."
    ),
    ScannerSon(
        model_id="scanner",
        name="AI Scanner (Son)",
        description="Specialist in hunting opportunities and scanning the market.",
        system_prompt="You are the Scanner, a specialized sub-agent. You are fast, precise, and obsessed with finding setups."
    ),
    TechnicalSon(
        model_id="technical",
        name="AI Technical (Son)",
        description="Expert chart analyst (TA). Focuses on price action and indicators.",
        system_prompt="You are the Technical Analyst. You speak in charts, levels, and indicators. You are cold and calculating."
    ),
    RugpullSon(
        model_id="rugpull",
        name="AI Rugpull (Son)",
        description="Security and Risk expert. Detects scams, honeypots, and rugpulls.",
        system_prompt="You are the Rugpull Detector. You are paranoid, skeptical, and protective. Trust nothing."
    )
]

def get_models():
    return [
        {
            "id": m.id,
            "name": m.name,
            "description": m.description
        } for m in models_list
    ]

def get_model_response(model_id: str, message: str, context: Optional[Dict] = None) -> Dict:
    model = next((m for m in models_list if m.id == model_id), models_list[0]) # Default to Mother
    
    response_text = model.generate_response(message, context)
    
    return {
        "response": response_text,
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model.name
    }
