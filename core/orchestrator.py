import time
from core.logger import log_event
from ml.scanner import MarketScanner
from trading.decision import DecisionEngine
from trading.execution import ExecutionEngine
from trading.position_manager import PositionManager

class Orchestrator:
    """
    SYSTEM CORE V3 (Ops-Enabled).
    Przekazuje signal_id przez cały pipeline.
    """

    def __init__(self, mode="PAPER"):
        self.mode = mode
        self.is_running = False
        
        self.scanner = MarketScanner(timeframe="1h")
        self.decision_engine = DecisionEngine(mode)
        self.execution_engine = ExecutionEngine(mode)
        self.position_manager = PositionManager(mode)
        
        log_event(f"ORCHESTRATOR READY [{self.mode}]", "INFO")

    def run_cycle(self):
        if not self.is_running: return

        # 1. RISK CHECK
        self.position_manager.manage_positions()
        is_safe, risk_msg = self.position_manager.check_global_risk()
        if not is_safe:
            log_event(f"LOCKED: {risk_msg}", "ERROR")
            return

        log_event("--- CYCLE START ---", "INFO")

        # 2. SCAN
        opportunities = self.scanner.scan()
        if opportunities.empty: return

        candidates = opportunities.head(3)
        
        for _, cand in candidates.iterrows():
            # 3. DECISION (Loguje event DECISION)
            approved, reason, size = self.decision_engine.evaluate_entry(cand, risk_status=is_safe)
            
            if approved:
                log_event(f"GO: {cand['symbol']} (${size})", "succ")
                
                # 4. EXECUTION (Przekazujemy signal_id, żeby spiąć logi)
                sig_id = cand.get('signal_id')
                res = self.execution_engine.execute_order(cand['symbol'], "BUY", size, signal_id=sig_id)
                
                if res['status'] == 'FILLED':
                    log_event(f"FILLED: {cand['symbol']}", "TRADE")
            else:
                log_event(f"SKIP {cand['symbol']}: {reason}", "WARN")

    def start_loop(self, interval=3600):
        self.is_running = True
        log_event("AUTO-PILOT ON", "succ")
        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        log_event("SYSTEM HALTED", "WARN")