import os
import json
import uuid
from datetime import datetime as dt


class EventLogger:
    """
    OPS ANALYTICS CORE.
    Append-only, daily-rotated JSONL event logger.
    Supports full trade lifecycle correlation via signal_id / trade_id.
    """

    LOG_DIR = os.path.join("assets", "ops_logs")
    SCHEMA_VERSION = "ops_event_v1"

    # ================================================================
    # INTERNALS
    # ================================================================

    @staticmethod
    def _ensure_dir():
        if not os.path.exists(EventLogger.LOG_DIR):
            os.makedirs(EventLogger.LOG_DIR)

    @staticmethod
    def _generate_signal_id(symbol: str, tf: str) -> str:
        """
        Deterministic + unique-enough signal ID.
        Example: SIG-20260110-BTCUSDT-1H-A1B2
        """
        ts = dt.utcnow().strftime("%Y%m%d%H%M%S")
        rand = uuid.uuid4().hex[:4].upper()
        sym = symbol.replace("/", "")
        return f"SIG-{ts}-{sym}-{tf.upper()}-{rand}"

    @staticmethod
    def _generate_trade_id(symbol: str) -> str:
        """
        Unique trade ID.
        Example: TRD-20260110-BTCUSDT-9F3C
        """
        ts = dt.utcnow().strftime("%Y%m%d%H%M%S")
        rand = uuid.uuid4().hex[:4].upper()
        sym = symbol.replace("/", "")
        return f"TRD-{ts}-{sym}-{rand}"

    @staticmethod
    def _write(event_type: str, payload: dict):
        EventLogger._ensure_dir()

        record = {
            "ts": dt.utcnow().isoformat(),
            "type": event_type,
            "schema": EventLogger.SCHEMA_VERSION,
            "data": payload
        }

        filename = f"{dt.utcnow().strftime('%Y-%m-%d')}.jsonl"
        filepath = os.path.join(EventLogger.LOG_DIR, filename)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[OPS LOGGER ERROR] {e}")

    # ================================================================
    # PUBLIC API
    # ================================================================

    @staticmethod
    def log_signal(symbol, tf, signal, conf, mqs, htf_trend, ev, reasons=None):
        signal_id = EventLogger._generate_signal_id(symbol, tf)

        EventLogger._write("SIGNAL", {
            "signal_id": signal_id,
            "symbol": symbol,
            "tf": tf,
            "signal": signal,
            "confidence": round(conf, 4),
            "mqs": mqs,
            "htf_trend": htf_trend,
            "ev": round(ev, 4),
            "reasons": reasons or []
        })

        return signal_id  # <-- IMPORTANT: return for chaining

    @staticmethod
    def log_decision(
        symbol,
        signal,
        approved,
        reason,
        size_usd,
        signal_id,
        checks=None,
        reasons=None
    ):
        EventLogger._write("DECISION", {
            "signal_id": signal_id,
            "symbol": symbol,
            "signal": signal,
            "approved": approved,
            "decision_reason": reason,
            "size_usd": round(size_usd, 2),
            "checks": checks or {},
            "reasons": reasons or []
        })

    @staticmethod
    def log_execution(
        symbol,
        side,
        entry_price,
        qty,
        cost,
        order_status,
        signal_id
    ):
        trade_id = EventLogger._generate_trade_id(symbol)

        EventLogger._write("EXECUTION", {
            "trade_id": trade_id,
            "signal_id": signal_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "qty": qty,
            "cost": round(cost, 2),
            "status": order_status,
            "env": os.getenv("TRADING_MODE", "PAPER")
        })

        return trade_id  # <-- IMPORTANT: return for position tracking

    @staticmethod
    def log_exit(
        symbol,
        entry_price,
        exit_price,
        pnl,
        reason,
        hold_time_min,
        trade_id
    ):
        EventLogger._write("EXIT", {
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "exit_reason": reason,
            "hold_time_min": round(hold_time_min, 1)
        })
