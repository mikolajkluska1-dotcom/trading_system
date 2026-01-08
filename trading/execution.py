from datetime import datetime
from trading.wallet import WalletManager
from core.logger import log_event
from data.feed import DataFeed
from core.event_logger import EventLogger  # <--- NOWOŚĆ

class ExecutionEngine:
    """
    MODUŁ WYKONAWCZY V3 (Ops-Connected).
    Generuje TRADE_ID i zapisuje go w aktywach.
    """

    def __init__(self, mode="PAPER"):
        self.mode = mode

    def execute_order(self, symbol, side, amount_usd, signal_id=None):
        wallet = WalletManager.get_wallet_data()
        
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
                balance = wallet.get('balance', 0.0)
                if balance < amount_usd:
                    return {'status': 'FAILED', 'reason': 'NSF'}
                
                # --- LOGOWANIE EXECUTION ---
                # Generujemy Trade ID powiązany z Signal ID
                trade_id = EventLogger.log_execution(
                    symbol=symbol,
                    side=side,
                    entry_price=fill_price,
                    qty=qty,
                    cost=amount_usd,
                    order_status="FILLED",
                    signal_id=signal_id
                )

                # Aktualizacja portfela
                wallet['balance'] = round(balance - amount_usd, 2)
                if 'assets' not in wallet: wallet['assets'] = []
                
                wallet['assets'].append({
                    "trade_id": trade_id,  # <--- ZAPISUJEMY ID
                    "sym": symbol,
                    "entry": fill_price,
                    "amt": qty,
                    "cost": amount_usd,
                    "ts": datetime.now().isoformat(), # ISO format lepszy dla analityki
                    "sl": fill_price * 0.96,
                    "tp": fill_price * 1.06
                })
                
                WalletManager.save_wallet_data(wallet)
                log_event(f"[PAPER] BOUGHT {symbol} (ID: {trade_id})", "TRADE")
                
                return {'status': 'FILLED', 'avg_price': fill_price, 'trade_id': trade_id}

        return {'status': 'FAILED', 'reason': 'BAD MODE'}

    def close_position(self, symbol, position_data, exit_price, reason="MANUAL"):
        if self.mode == "PAPER":
            wallet = WalletManager.get_wallet_data()
            qty = position_data['amt']
            cost = position_data['cost']
            trade_id = position_data.get('trade_id', 'UNKNOWN') # Pobieramy ID
            
            revenue = qty * exit_price
            pnl = revenue - cost
            
            wallet['balance'] = round(wallet.get('balance', 0) + revenue, 2)
            
            # Obliczanie czasu trwania
            hold_min = 0
            try:
                entry_ts = datetime.fromisoformat(position_data['ts'])
                hold_min = (datetime.now() - entry_ts).total_seconds() / 60
            except: pass

            # --- LOGOWANIE WYJŚCIA ---
            EventLogger.log_exit(
                symbol=symbol,
                entry_price=position_data['entry'],
                exit_price=exit_price,
                pnl=pnl,
                reason=reason,
                hold_time_min=hold_min,
                trade_id=trade_id
            )
            
            # Historia w portfelu (dla szybkiego UI)
            if 'history' not in wallet: wallet['history'] = []
            wallet['history'].append({
                "date": datetime.now().strftime('%Y-%m-%d'),
                "ts": datetime.now().isoformat(),
                "symbol": symbol,
                "action": "SELL",
                "pnl_val": round(pnl, 2),
                "desc": f"EXIT {symbol} | PnL: ${pnl:.2f}"
            })
            
            WalletManager.save_wallet_data(wallet)
            return True
            
        return False