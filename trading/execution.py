from datetime import datetime
from trading.wallet import WalletManager
from core.logger import log_event
from data.feed import DataFeed
from core.event_logger import EventLogger

class ExecutionEngine:
    """
    MODUŁ WYKONAWCZY V3.0 (SQLite Powered).
    """

    def __init__(self, mode="PAPER"):
        self.mode = mode

    def execute_order(self, symbol, side, amount_usd, signal_id=None):
        # 1. Price Check
        try:
            ticker_data = DataFeed.get_market_data(symbol, "1m", limit=1)
            if ticker_data.empty: return {'status': 'FAILED', 'reason': 'NO DATA'}
            fill_price = ticker_data['close'].iloc[-1]
        except Exception as e:
            return {'status': 'FAILED', 'reason': str(e)}

        qty = amount_usd / fill_price

        # 2. PAPER EXECUTION
        if self.mode == "PAPER":
            if side == "BUY":
                balance = WalletManager.get_balance()
                if balance < amount_usd:
                    return {'status': 'FAILED', 'reason': 'NSF'}

                # --- LOGOWANIE EXECUTION ---
                trade_id = EventLogger.log_execution(
                    symbol=symbol,
                    side=side,
                    entry_price=fill_price,
                    qty=qty,
                    cost=amount_usd,
                    order_status="FILLED",
                    signal_id=signal_id
                )

                # Aktualizacja portfela w bazie danych
                WalletManager.update_balance(round(balance - amount_usd, 2))
                WalletManager.record_trade(
                    symbol=symbol,
                    side=side,
                    price=fill_price,
                    amount=qty,
                    cost=amount_usd,
                    signal_id=signal_id,
                    reason="AI_SIGNAL"
                )

                log_event(f"[PAPER] BOUGHT {symbol} (ID: {trade_id})", "TRADE")
                return {'status': 'FILLED', 'avg_price': fill_price, 'trade_id': trade_id}

        return {'status': 'FAILED', 'reason': 'BAD MODE'}

    def close_position(self, symbol, position_data, exit_price, reason="MANUAL"):
        if self.mode == "PAPER":
            qty = position_data['size']
            cost = position_data['entry'] * qty

            revenue = qty * exit_price
            pnl = revenue - cost

            # Aktualizacja balansu
            current_balance = WalletManager.get_balance()
            WalletManager.update_balance(round(current_balance + revenue, 2))

            # Record sell trade
            WalletManager.record_trade(
                symbol=symbol,
                side="SELL",
                price=exit_price,
                amount=qty,
                cost=revenue,
                reason=reason
            )

            log_event(f"[PAPER] SOLD {symbol} (PnL: ${pnl:.2f})", "TRADE")
            return True

        return False
