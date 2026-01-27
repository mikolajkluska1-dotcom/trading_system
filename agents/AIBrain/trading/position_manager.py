from datetime import datetime, timedelta
from agents.AIBrain.trading.wallet import WalletManager
from agents.AIBrain.trading.execution import ExecutionEngine
from data.feed import DataFeed
from agents.Database.core.logger import log_event

class PositionManager:
    """
    ZARZĄDCA POZYCJI V3.0 (SQLite Powered).
    """

    def __init__(self, mode="PAPER"):
        self.mode = mode
        self.executor = ExecutionEngine(mode)
        self.MAX_DAILY_LOSS_USD = -500.0
        self.MAX_HOLD_HOURS = 24
        self.kill_switch_active = False

    def get_daily_pnl(self):
        wallet = WalletManager.get_wallet_data()
        today_str = datetime.now().strftime('%Y-%m-%d')
        # Uproszczone liczenie PnL z historii (można rozbudować o kolumnę pnl w DB)
        return 0.0

    def check_global_risk(self):
        if self.kill_switch_active: return False, "KILL SWITCH"
        return True, "OK"

    def manage_positions(self):
        is_safe, _ = self.check_global_risk()
        if not is_safe: return

        wallet = WalletManager.get_wallet_data()
        assets = wallet.get('assets', [])
        if not assets: return

        for pos in assets:
            symbol = pos['sym']

            # Pobieramy cenę
            try:
                df = DataFeed.get_market_data(symbol, "1m", limit=1)
                curr_price = df['close'].iloc[-1]
            except: continue

            # Warunki wyjścia (Uproszczone: SL/TP)
            reason = ""
            if curr_price <= pos.get('entry', 0) * 0.96: reason = "STOP LOSS"
            elif curr_price >= pos.get('entry', 0) * 1.06: reason = "TAKE PROFIT"

            if reason:
                log_event(f"EXIT {symbol}: {reason}", "WARN")
                self.executor.close_position(symbol, pos, curr_price, reason)
