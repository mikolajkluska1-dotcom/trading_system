"""
AIBrain v2.0 - Base Agent
Base class for all child agents
"""

class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.last_analysis = {'signal': 'NEUTRAL', 'score': 0.0}

    async def analyze(self, market_data):
        """Override in child classes"""
        return {'signal': 'NEUTRAL', 'score': 0.0}
    
    def get_signal(self):
        """Get the last analysis result"""
        return self.last_analysis
