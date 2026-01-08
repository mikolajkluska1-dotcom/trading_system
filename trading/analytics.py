import pandas as pd
import numpy as np
from trading.wallet import WalletManager

class TradeAnalytics:
    """
    MODUŁ ANALITYCZNY (EX-POST).
    Liczy KPI: WinRate, Expectancy, Drawdown, Profit Factor.
    """

    @staticmethod
    def generate_report():
        wallet = WalletManager.get_wallet_data()
        history = wallet.get('history', [])
        
        # Filtrujemy tylko zamknięte transakcje (gdzie jest pnl_val)
        trades = [h for h in history if isinstance(h, dict) and 'pnl_val' in h]
        
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0,
                "net_profit": 0.0
            }

        df = pd.DataFrame(trades)
        
        # 1. Podstawowe metryki
        total_trades = len(df)
        wins = df[df['pnl_val'] > 0]
        losses = df[df['pnl_val'] <= 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        net_profit = df['pnl_val'].sum()
        
        # 2. Profit Factor & Expectancy
        gross_profit = wins['pnl_val'].sum()
        gross_loss = abs(losses['pnl_val'].sum())
        
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')
        
        avg_win = wins['pnl_val'].mean() if not wins.empty else 0
        avg_loss = losses['pnl_val'].mean() if not losses.empty else 0
        
        # EV = (Win% * AvgWin) - (Loss% * AvgLoss)
        win_dec = win_rate / 100
        expectancy = (win_dec * avg_win) + ((1 - win_dec) * avg_loss)

        # 3. Max Drawdown (Symulacja krzywej kapitału)
        df['equity_curve'] = df['pnl_val'].cumsum()
        df['peak'] = df['equity_curve'].cummax()
        df['drawdown'] = df['equity_curve'] - df['peak']
        max_dd = df['drawdown'].min()

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": profit_factor,
            "expectancy": round(expectancy, 2), # Oczekiwany zysk na 1 trade
            "max_drawdown": round(max_dd, 2),
            "net_profit": round(net_profit, 2)
        }