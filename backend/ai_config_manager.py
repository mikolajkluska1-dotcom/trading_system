import json
import os
import logging

CONFIG_PATH = os.path.join("data", "ai_config.json")

DEFAULT_CONFIG = {
    "min_confidence": 0.60,
    "risk_mode": "BALANCED", # CONSERVATIVE, BALANCED, DEGEN
    "sentiment_weight": 50.0, # 0-100 impact of external sentiment
    "max_open_positions": 3,
    "auto_trade_enabled": False,
    "confirmation_required": True, # Human-in-the-loop
    "volatility_filter": True,
    "news_impact_enabled": True,
    "copy_trading_enabled": True,
    "whale_trust_factor": 0.5
}

logger = logging.getLogger("AI_CONFIG")

class AIConfigManager:
    """
    Manages persistent AI settings.
    """
    @staticmethod
    def load_config():
        if not os.path.exists(CONFIG_PATH):
            AIConfigManager.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
        
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                for k, v in DEFAULT_CONFIG.items():
                    if k not in config:
                        config[k] = v
                return config
        except Exception as e:
            logger.error(f"Failed to load AI config: {e}")
            return DEFAULT_CONFIG

    @staticmethod
    def save_config(config):
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Failed to save AI config: {e}")
            return False

    @staticmethod
    def update_config(updates):
        current = AIConfigManager.load_config()
        # Validate and update
        for k, v in updates.items():
            if k in DEFAULT_CONFIG:
                current[k] = v
        
        AIConfigManager.save_config(current)
        return current

    @staticmethod
    def get_defaults():
        return {
            "min_confidence": 0.60,
            "risk_mode": "BALANCED", 
            "sentiment_weight": 50.0,
            "max_open_positions": 3,
            "auto_trade_enabled": False,
            "confirmation_required": True,
            "volatility_filter": True,
            "news_impact_enabled": True,
            "copy_trading_enabled": True,    # New: Master toggle for whale copy
            "whale_trust_factor": 0.5        # New: 0.0-1.0 influence of whale signals
        }
