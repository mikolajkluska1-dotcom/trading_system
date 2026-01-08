from datetime import datetime, time
from trading.wallet import WalletManager
from trading.capital import CapitalGuard
from data.feed import DataFeed
from core.event_logger import EventLogger  # <--- NOWOŚĆ

class DecisionEngine:
    """
    STRAŻNIK RYZYKA V4 (Ops-Connected).
    Loguje każdą decyzję (TAK/NIE) do EventLoggera.
    """

    def __init__(self, mode="PAPER"):
        self.mode = mode
        self.max_concurrent_positions = 3
        self.min_confidence = 0.60
        
        self.COOLDOWN_MINUTES = 60
        self.SESSION_START = time(7, 0)  
        self.SESSION_END = time(21, 0)   

    def _check_session(self):
        now = datetime.utcnow().time()
        return self.SESSION_START <= now <= self.SESSION_END

    def _check_cooldown(self, symbol):
        wallet = WalletManager.get_wallet_data()
        history = wallet.get('history', [])
        
        last_trade_ts = None
        for trade in reversed(history):
            if isinstance(trade, dict) and trade.get('symbol') == symbol:
                last_trade_ts = trade.get('ts')
                break
        
        if last_trade_ts:
            try:
                # Obsługa formatu ISO z EventLoggera lub prostego czasu
                if "T" in last_trade_ts:
                    last_dt = datetime.fromisoformat(last_trade_ts)
                else:
                    # Fallback dla starego formatu
                    last_dt = datetime.strptime(last_trade_ts, "%H:%M:%S")
                    now = datetime.now()
                    last_dt = last_dt.replace(year=now.year, month=now.month, day=now.day)
                
                diff = (datetime.now() - last_dt).total_seconds() / 60
                if diff < self.COOLDOWN_MINUTES:
                    return False, f"COOLDOWN ({int(diff)}m)"
            except: pass
                
        return True, "OK"

    def evaluate_entry(self, candidate, risk_status=True):
        """
        Zwraca: (approved, reason, size_usd)
        Oraz LOGUJE decyzję.
        """
        symbol = candidate['symbol']
        signal = candidate['signal']
        signal_id = candidate.get('signal_id', 'MANUAL') # Pobieramy ID ze skanera
        
        reasons = []
        checks = {}
        approved = False
        final_reason = "UNKNOWN"
        size_usd = 0.0

        # --- LOGIKA DECYZYJNA ---
        try:
            # 1. Global Risk
            checks['global_risk'] = risk_status
            if not risk_status:
                final_reason = "KILL_SWITCH"
                raise StopIteration

            # 2. Session
            sess_ok = self._check_session()
            checks['session'] = sess_ok
            if not sess_ok:
                final_reason = "OUT_OF_SESSION"
                raise StopIteration

            # 3. Cooldown
            cool_ok, cool_msg = self._check_cooldown(symbol)
            checks['cooldown'] = cool_ok
            if not cool_ok:
                final_reason = cool_msg
                raise StopIteration

            # 4. Portfolio
            wallet = WalletManager.get_wallet_data()
            curr_assets = [a['sym'] for a in wallet.get('assets', [])]
            if symbol in curr_assets:
                final_reason = "ALREADY_IN_POS"
                raise StopIteration
            
            if len(curr_assets) >= self.max_concurrent_positions:
                final_reason = "MAX_SLOTS_FULL"
                raise StopIteration

            # 5. Price & Size
            price = candidate.get('current_price', 0)
            if price <= 0:
                try:
                    df = DataFeed.get_market_data(symbol, "1m", limit=1)
                    price = df['close'].iloc[-1]
                except:
                    final_reason = "NO_PRICE_DATA"
                    raise StopIteration

            balance = wallet.get('balance', 0)
            mqs = candidate.get('mqs', 50)
            
            size, _ = CapitalGuard.calculate_position_size(balance, price, mqs)
            if size < 10:
                final_reason = "SIZE_TOO_SMALL"
                raise StopIteration

            # SUKCES
            approved = True
            final_reason = "APPROVED"
            size_usd = size

        except StopIteration:
            pass

        # --- LOGOWANIE ZDARZENIA ---
        EventLogger.log_decision(
            symbol=symbol,
            signal=signal,
            approved=approved,
            reason=final_reason,
            size_usd=size_usd,
            signal_id=signal_id,
            checks=checks
        )
        DecisionCore = DecisionEngine


        return approved, final_reason, size_usd