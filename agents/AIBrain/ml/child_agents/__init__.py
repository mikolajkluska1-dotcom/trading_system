"""
AIBrain v4.0 - Child Agents Module (MEGA UPGRADE)
"""
from .base_agent import BaseAgent
from .portfolio_manager_agent import PortfolioManagerAgent
from .scanner_agent import ScannerAgent
from .technician_agent import TechnicianAgent
from .whale_watcher_agent import WhaleWatcherAgent
from .sentiment_agent import SentimentAgent
from .rugpull_detector_agent import RugpullDetectorAgent
from .trend_agent import TrendAgent
from .mtf_agent import MTFAgent
from .correlation_agent import CorrelationAgent

__all__ = [
    'BaseAgent',
    'PortfolioManagerAgent',
    'ScannerAgent',
    'TechnicianAgent',
    'WhaleWatcherAgent',
    'SentimentAgent',
    'RugpullDetectorAgent',
    'TrendAgent',
    'MTFAgent',
    'CorrelationAgent',
]

