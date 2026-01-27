# core/auto_rotator.py
from agents.AIBrain.ml.scanner import MarketScanner
from datetime import datetime

class AutoRotator:
    """
    Manualny moduł do jednorazowego uruchomienia skanu rynku.
    (Nie działa w pętli - manualny refresh przez UI)
    """

    @staticmethod
    def run_scan(timeframe="1h"):
        print(f"[{datetime.utcnow()}] Starting market scan ({timeframe})...")
        scanner = MarketScanner(timeframe)
        ranked = scanner.scan()
        print(f"Scan completed: {len(ranked)} assets analyzed.")
        return ranked
