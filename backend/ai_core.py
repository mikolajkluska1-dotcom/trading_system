# backend/ai_core.py
"""
REDLINE AI CORE — GEN 2
Stateful, Explainable, Auto-Trading Ready

ML logic lives in /ml
This file orchestrates decisions only.
"""

import time
import threading
from datetime import datetime, timedelta

from ml.scanner import MarketScanner
from ml.brain import DeepBrain
from ml.regime import MarketRegime

from trading.decision import DecisionEngine
from trading.execution import ExecutionEngine
from trading.wallet import WalletManager

from core.logger import log_event
from core.event_logger import EventLogger


class RedlineAICore:
    """
    Central AI Orchestrator (GEN-2)
    """

    def __init__(self, mode="PAPER", timeframe="1h"):
        self.mode = mode
        self.timeframe = timeframe

        # ML
        self.scanner = MarketScanner(timeframe=timeframe)
        self.brain = DeepBrain()

        # Trading
        self.decision = DecisionEngine(mode=mode)
        self.executor = ExecutionEngine(mode=mode)

        # ===== AI STATE =====
        self.state = {
            "mode": "SEMI",               # MANUAL | SEMI | AUTO
            "running": False,
            "last_cycle": None,
            "ai_score": 0.0,
            "cooldowns": {},              # symbol -> datetime
            "last_signals": {},           # symbol -> signal
            "errors": 0
        }

        log_event(f"AI CORE GEN-2 INIT [{mode} | {timeframe}]", "INFO")

    # =====================================================
    # PUBLIC CONTROL
    # =====================================================
    def set_mode(self, mode: str):
        if mode not in {"MANUAL", "SEMI", "AUTO"}:
            raise ValueError("Invalid AI mode")
        self.state["mode"] = mode
        log_event(f"AI MODE SET → {mode}", "WARN")

    def get_state(self):
        wallet = WalletManager.get_wallet_data()
        return {
            **self.state,
            "balance": wallet.get("balance", 0.0),
            "positions": wallet.get("assets", []),
        }

    # =====================================================
    # MAIN AI CYCLE
    # =====================================================
    def run_once(self):
        """
        One full AI decision cycle.
        """
        self.state["last_cycle"] = datetime.utcnow().isoformat()
        log_event("AI CYCLE START", "INFO")

        try:
            df = self.scanner.scan()
            if df.empty:
                log_event("SCAN EMPTY", "WARN")
                return []

            results = []

            for _, row in df.head(5).iterrows():
                symbol = row["symbol"]

                # ===== COOLDOWN CHECK =====
                cd = self.state["cooldowns"].get(symbol)
                if cd and datetime.utcnow() < cd:
                    results.append({
                        "symbol": symbol,
                        "status": "COOLDOWN"
                    })
                    continue

                # ===== SIGNAL LOG =====
                signal_id = EventLogger.log_signal(
                    symbol=symbol,
                    tf=self.timeframe,
                    signal=row.get("signal"),
                    conf=row.get("confidence", 0),
                    mqs=row.get("mqs", 0),
                    htf_trend=row.get("htf_trend", "NEUTRAL"),
                    ev=row.get("ev", 0),
                )

                candidate = {
                    "symbol": symbol,
                    "signal": row.get("signal"),
                    "conf": row.get("confidence"),
                    "mqs": row.get("mqs"),
                    "current_price": row.get("price"),
                    "reasons": row.get("reasons", [])
                }

                # ===== DECISION =====
                approved, reason, size = self.decision.evaluate_entry(candidate)

                EventLogger.log_decision(
                    symbol=symbol,
                    signal=candidate["signal"],
                    approved=approved,
                    reason=reason,
                    size_usd=size,
                    signal_id=signal_id
                )

                if not approved:
                    results.append({
                        "symbol": symbol,
                        "status": "REJECTED",
                        "reason": reason
                    })
                    continue

                # ===== SEMI MODE =====
                if self.state["mode"] == "SEMI":
                    results.append({
                        "symbol": symbol,
                        "status": "APPROVED_WAITING",
                        "size": size
                    })
                    continue

                # ===== AUTO MODE =====
                if self.state["mode"] == "AUTO":
                    exec_res = self.executor.execute_order(
                        symbol=symbol,
                        side="BUY",
                        amount_usd=size,
                        signal_id=signal_id
                    )

                    if exec_res.get("status") == "FILLED":
                        EventLogger.log_execution(
                            symbol=symbol,
                            side="BUY",
                            entry_price=exec_res.get("avg_price"),
                            qty=exec_res.get("qty"),
                            cost=exec_res.get("cost"),
                            order_status="FILLED",
                            signal_id=signal_id
                        )

                        # cooldown 1 cycle
                        self.state["cooldowns"][symbol] = (
                            datetime.utcnow() + timedelta(minutes=30)
                        )

                    results.append(exec_res)

            log_event("AI CYCLE END", "INFO")
            return results

        except Exception as e:
            self.state["errors"] += 1
            log_event(f"AI CORE ERROR: {e}", "CRITICAL")
            return []

    # =====================================================
    # AUTO MODE LOOP
    # =====================================================
    def _auto_loop(self, interval_sec: int):
        log_event("AI AUTO LOOP STARTED", "WARN")
        self.state["running"] = True

        while self.state["running"]:
            self.run_once()
            time.sleep(interval_sec)

    def start_auto(self, interval_sec: int = 3600):
        if self.state["running"]:
            return
        self.set_mode("AUTO")
        thread = threading.Thread(
            target=self._auto_loop,
            args=(interval_sec,),
            daemon=True
        )
        thread.start()

    def stop_auto(self):
        self.state["running"] = False
        self.set_mode("SEMI")
        log_event("AI AUTO LOOP STOPPED", "WARN")
