from datetime import datetime, timedelta
from trading.wallet import WalletManager
from trading.execution import ExecutionEngine
from data.feed import DataFeed
from core.logger import log_event

class PositionManager:
    """
    ZARZĄDCA POZYCJI V3.
    Kompatybilny z nowym formatem daty w portfelu.
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
        return sum(e.get('pnl_val', 0) for e in wallet.get('history', []) if e.get('date') == today_str)

    def check_global_risk(self):
        if self.kill_switch_active: return False, "KILL SWITCH"
        if self.get_daily_pnl() <= self.MAX_DAILY_LOSS_USD:
            self.kill_switch_active = True
            return False, "DAILY LOSS LIMIT"
        return True, "OK"

    def manage_positions(self):
        is_safe, _ = self.check_global_risk()
        if not is_safe: return

        wallet = WalletManager.get_wallet_data()
        assets = wallet.get('assets', [])
        if not assets: return

        dirty = False
        for pos in assets[:]:
            symbol = pos['sym']
            
            # Obsługa daty (ISO vs Legacy)
            try:
                if "T" in pos['ts']:
                    entry_dt = datetime.fromisoformat(pos['ts'])
                else:
                    entry_dt = datetime.strptime(pos['ts'], "%Y-%m-%d %H:%M:%S")
            except:
                entry_dt = datetime.now()

            # Pobieramy cenę
            try:
                df = DataFeed.get_market_data(symbol, "1m", limit=1)
                curr_price = df['close'].iloc[-1]
            except: continue

            # Warunki wyjścia
            reason = ""
            if curr_price <= pos.get('sl', 0): reason = "STOP LOSS"
            elif curr_price >= pos.get('tp', 999999): reason = "TAKE PROFIT"
            elif (datetime.now() - entry_dt) > timedelta(hours=self.MAX_HOLD_HOURS): reason = "TIME STOP"

            if reason:
                log_event(f"EXIT {symbol}: {reason}", "WARN")
                if self.executor.close_position(symbol, pos, curr_price, reason):
                    assets.remove(pos)
                    dirty = True
        
        if dirty:
            wallet['assets'] = assets
            WalletManager.save_wallet_data(wallet)