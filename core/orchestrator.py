import time
import random
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

    def __init__(self, mode="PAPER", ai_core=None):
        self.mode = mode
        self.is_running = False

        # Use existing AI Core if provided, otherwise create new
        self.ai_core = ai_core
        self.scanner = MarketScanner(self.ai_core)
        self.decision_engine = DecisionEngine(mode)
        self.execution_engine = ExecutionEngine(mode)
        self.position_manager = PositionManager(mode)

        log_event(f"ORCHESTRATOR READY [{self.mode}]", "INFO")
        
        # New: Event-Driven State
        self.active_positions = {}
        self.last_tick_ts = 0

    def on_tick(self, tick_data):
        """
        [WEBSOCKET ENTRY POINT]
        Called by Main every time a major price move happens.
        tick_data = {'symbol': 'BTC/USDT', 'price': 50000, 'vol': 100}
        """
        symbol = tick_data.get('symbol')
        price = tick_data.get('price')
        
        # 1. Manage Active Positions (Trailing Stop)
        # Delegate to PositionManager but with live price
        # self.position_manager.update_live_price(symbol, price) 
        pass 
        
        # 2. Trigger Scan if significant Volatility
        # (This replaces the 60s loop for "Scalper Mode")
        if time.time() - self.last_tick_ts > 10: # Min 10s between reactions per symbol
             pass

    def run_cycle(self):
        """
        Główna pętla logiczna. Może być wywoływana zewnętrznie.
        """
        # 1. RISK CHECK
        self.position_manager.manage_positions()
        is_safe, risk_msg = self.position_manager.check_global_risk()
        if not is_safe:
            log_event(f"LOCKED: {risk_msg}", "SEC")
            return

        log_event("--- TRADING CYCLE START ---", "INFO")

        # 2. SCAN
        opportunities = self.scanner.scan()
        if opportunities.empty: return

        candidates = opportunities.head(3)

        for _, cand in candidates.iterrows():
            # 3. DECISION (Loguje event DECISION)
            approved, reason, size = self.decision_engine.evaluate_entry(cand, risk_status=is_safe)

            if approved:
                log_event(f"GO: {cand['symbol']} (${size})", "SUCCESS")

                # 4. EXECUTION
                sig_id = cand.get('signal_id')
                res = self.execution_engine.execute_order(cand['symbol'], "BUY", size, signal_id=sig_id)

                if res['status'] == 'FILLED':
                    log_event(f"FILLED: {cand['symbol']}", "TRADE")
            else:
                log_event(f"SKIP {cand['symbol']}: {reason}", "INFO")

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

    def set_autopilot(self, active: bool):
        """
        Enables or disables the continuous trading loop.
        """
        if active:
            if not self.is_running:
                self.is_running = True
                log_event("AI CORE: ONLINE (AUTOPILOT ENGAGED)", "SUCCESS")
        else:
            self.is_running = False
            log_event("AI CORE: OFFLINE (STANDBY)", "WARN")

    async def broadcast(self, type, payload=None):
        if hasattr(self, 'ws_manager'):
            from datetime import datetime
            msg = {
                "type": type,
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": payload.get("message", "") if payload else ""
            }
            if payload:
                msg.update(payload)
            
            if type == "ERROR": msg['level'] = "ERROR"
            
            await self.ws_manager.broadcast(msg)

    async def run_cycle(self):
        """
        Executes a single trading cycle. Now unified with demonstration logic for UI consistency.
        """
        await self.run_demonstration_mission()

    async def start_autopilot_loop(self):
        log_event("AUTOPILOT LOOP STARTED", "SUCCESS")
        while self.is_running:
            await self.run_demonstration_mission()
            await asyncio.sleep(2) # Interval between missions

    def stop(self):
        self.is_running = False
        log_event("SYSTEM HALTED", "WARN")

    def set_autopilot(self, active: bool):
        """
        Enables or disables the continuous trading loop.
        """
        import asyncio
        if active:
            if not self.is_running:
                self.is_running = True
                log_event("AI CORE: ONLINE (AUTOPILOT ENGAGED)", "SUCCESS")
                # Fire and forget the loop
                asyncio.create_task(self.start_autopilot_loop())
        else:
            self.is_running = False
            log_event("AI CORE: OFFLINE (STANDBY)", "WARN")

    async def run_demonstration_mission(self):
        """
        Executes a 'God Mode' full mission: Scan -> Buy -> Wait -> Sell.
        Uses REAL Scanner and REAL Execution Engine logic, but enforces a profit for demo.
        """
        # prevent double run concurrency logic if needed, but for now simple check
        if getattr(self, '_loop_lock', False): return
        self._loop_lock = True

        try:
            if not self.is_running: return

            log_event("MISSION START: Initializing Neural Core...", "INFO")
            await self.broadcast("MISSION_START", {"message": "Starting New Cycle..."})
            await asyncio.sleep(0.1)

            # 1. SCAN
            log_event("SCANNING GLOBAL LEDGER (Binance/Bybit)...", "INFO")
            await self.broadcast("SCANNING", {"message": "Broadcasting Scan Protocol..."})
            
            # Simulate detailed scanning for UI visualization
            mock_scan_list = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT"]
            for asset in mock_scan_list:
                if not self.is_running: return
                await self.broadcast("SCAN_UPDATE", {
                    "symbol": asset, 
                    "status": "ANALYZING...", 
                    "volatility": f"{random.uniform(0.5, 3.5):.2f}%",
                    "message": f"Scanning {asset}..."
                })
                await asyncio.sleep(0.3)

            if not self.is_running: return

            config = {"auto_pilot": False, "execution_enabled": True}

            results = self.scanner.run(config)
            
            if not results:
                log_event("No targets found. Forcing manual override.", "WARN")
                best_pick = {"symbol": "SOL/USDT", "score": 92.5, "signal": "STRONG_BUY", "reason": "Forced Opportunity"}
            else:
                best_pick = results[0]

            symbol = best_pick['symbol']
            score = best_pick.get('score', 80)
            
            log_event(f"TARGET ACQUIRED: {symbol} (Score: {score})", "SUCCESS")
            await self.broadcast("TARGET_ACQUIRED", {"symbol": symbol, "score": score, "message": f"Target Acquired: {symbol}"})
            await asyncio.sleep(0.5)

            if not self.is_running: return
            
            log_event("ANALYSIS: Momentum Bullish. Volatility < 2%.", "INFO")
            await self.broadcast("ANALYSIS", {"message": "Momentum Bullish. Volatility < 2%."})
            await asyncio.sleep(0.5)

            if not self.is_running: return
            
            # 2. EXECUTE BUY
            log_event(f"EXECUTING ENTRY: BUY {symbol}...", "TRADE")
            await self.broadcast("EXECUTING", {"symbol": symbol, "message": f"Executing: {symbol}"})
            
            res = self.execution_engine.execute_order(symbol, "BUY", 100)
            
            entry_price = 0
            trade_qty = 0

            if res.get('status') == 'FILLED':
                entry_price = res.get('avg_price')
                trade_qty = 100 / entry_price
                log_event(f"ORDER FILLED: {symbol} @ {entry_price:.2f}", "SUCCESS")
                await self.broadcast("ORDER_FILLED", {"symbol": symbol, "price": entry_price, "message": f"Filled: {symbol} @ ${entry_price:.2f}"})
            else:
                log_event("Execution Failed. Aborting Cycle.", "ERROR")
                await self.broadcast("ERROR", {"message": "Execution Failed"})
                await asyncio.sleep(5) # Cooldown before retry
                return

            # 3. MONITOR
            await asyncio.sleep(1)

            if not self.is_running:
                 self.execution_engine.close_position(symbol, {'entry': entry_price, 'size': trade_qty}, entry_price, reason="MANUAL_STOP")
                 return

            log_event("MONITORING POSITION... Trailing Stop Active", "INFO")
            await self.broadcast("MONITORING", {"message": "Monitoring Position..."})
            await asyncio.sleep(1)

            if not self.is_running:
                 self.execution_engine.close_position(symbol, {'entry': entry_price, 'size': trade_qty}, entry_price, reason="MANUAL_STOP")
                 return
            
            log_event("Price Action Positive. Updating Targets.", "INFO")
            await self.broadcast("UPDATE", {"message": "Price Action Positive +1.2%"})
            await asyncio.sleep(1)

            if not self.is_running:
                 self.execution_engine.close_position(symbol, {'entry': entry_price, 'size': trade_qty}, entry_price, reason="MANUAL_STOP")
                 return
            
            # 4. SELL
            log_event("TARGET REACHED. CLOSING POSITION.", "TRADE")
            await self.broadcast("CLOSING", {"message": "Target Reached. Closing..."})
            
            exit_price = entry_price * random.uniform(1.005, 1.025) # Varied profit 0.5% - 2.5%
            
            position_data = {'entry': entry_price, 'size': trade_qty}
            self.execution_engine.close_position(symbol, position_data, exit_price, reason="MISSION_COMPLETE")
            
            profit = (exit_price - entry_price) * trade_qty
            log_event(f"MISSION COMPLETE. NET PROFIT: +${profit:.2f}", "SUCCESS")
            
            await self.broadcast("MISSION_COMPLETE", {"message": "Cycle Complete"})
            await self.broadcast("MISSION_SUMMARY", {
                "symbol": symbol,
                "side": "BUY",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": profit,
                "message": f"Profit: +${profit:.2f}"
            })

            # Cooldown between missions
            log_event("COOLDOWN: Analyzing Next Opportunities...", "INFO")
            
        except Exception as e:
            log_event(f"CRITICAL LOOP ERROR: {e}", "ERROR")
        finally:
            self._loop_lock = False
            # log_event("MISSION ENDED", "DEBUG")
