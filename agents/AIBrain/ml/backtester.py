"""
AIBrain v4.0 - MEGA UPGRADE
Backtester Engine - STRATEGY VALIDATOR
Symuluje trading na historycznych danych
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
import json


class Trade:
    """Represents a single trade"""
    def __init__(self, entry_time, entry_price, direction, size_usd=100):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction  # 'LONG' or 'SHORT'
        self.size_usd = size_usd
        
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.is_open = True
    
    def close(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.is_open = False
        
        if self.direction == 'LONG':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.pnl = self.size_usd * self.pnl_pct
        return self.pnl
    
    def to_dict(self):
        return {
            'entry_time': str(self.entry_time),
            'entry_price': self.entry_price,
            'exit_time': str(self.exit_time) if self.exit_time else None,
            'exit_price': self.exit_price,
            'direction': self.direction,
            'size_usd': self.size_usd,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct * 100
        }


class BacktestResult:
    """Contains all backtest results and metrics"""
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.total_pnl = 0.0
        self.total_return_pct = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.avg_trade_pnl = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
    def calculate_metrics(self, initial_capital: float = 10000):
        """Calculate all performance metrics"""
        if not self.trades:
            return
        
        closed_trades = [t for t in self.trades if not t.is_open]
        self.total_trades = len(closed_trades)
        
        if self.total_trades == 0:
            return
        
        # Win/Loss stats
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # PnL stats
        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
        
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        self.total_pnl = sum(t.pnl for t in closed_trades)
        self.total_return_pct = (self.total_pnl / initial_capital) * 100
        
        # Average stats
        self.avg_trade_pnl = self.total_pnl / self.total_trades
        self.avg_win = total_wins / len(wins) if wins else 0
        self.avg_loss = total_losses / len(losses) if losses else 0
        
        # Extremes
        self.largest_win = max(t.pnl for t in wins) if wins else 0
        self.largest_loss = min(t.pnl for t in losses) if losses else 0
        
        # Equity curve and drawdown
        equity = initial_capital
        peak = initial_capital
        self.equity_curve = [initial_capital]
        self.drawdown_curve = [0]
        
        for trade in closed_trades:
            equity += trade.pnl
            self.equity_curve.append(equity)
            
            if equity > peak:
                peak = equity
            
            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0
            self.drawdown_curve.append(dd_pct * 100)
            
            if dd > self.max_drawdown:
                self.max_drawdown = dd
                self.max_drawdown_pct = dd_pct * 100
        
        # Sharpe Ratio (simplified - daily returns assumed)
        returns = [t.pnl_pct for t in closed_trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            self.sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # Sortino (downside deviation only)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                self.sortino_ratio = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    def to_dict(self):
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'profit_factor': round(self.profit_factor, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'avg_trade_pnl': round(self.avg_trade_pnl, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2)
        }
    
    def print_report(self):
        """Print formatted backtest report"""
        print("\n" + "="*60)
        print("📊 BACKTEST REPORT")
        print("="*60)
        print(f"Total Trades:      {self.total_trades}")
        print(f"Win Rate:          {self.win_rate*100:.1f}%")
        print(f"Profit Factor:     {self.profit_factor:.2f}")
        print("-"*60)
        print(f"Total PnL:         ${self.total_pnl:.2f}")
        print(f"Total Return:      {self.total_return_pct:.2f}%")
        print(f"Max Drawdown:      {self.max_drawdown_pct:.2f}%")
        print("-"*60)
        print(f"Sharpe Ratio:      {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:     {self.sortino_ratio:.2f}")
        print("-"*60)
        print(f"Avg Trade:         ${self.avg_trade_pnl:.2f}")
        print(f"Avg Win:           ${self.avg_win:.2f}")
        print(f"Avg Loss:          ${self.avg_loss:.2f}")
        print(f"Largest Win:       ${self.largest_win:.2f}")
        print(f"Largest Loss:      ${self.largest_loss:.2f}")
        print("="*60 + "\n")


class Backtester:
    """
    Backtester Engine for AIBrain strategies
    
    Usage:
        backtester = Backtester(initial_capital=10000)
        backtester.load_data("R:/Redline_Data/bulk_data/klines/1h")
        
        # Define signal function
        def my_strategy(df, i):
            # Return 'BUY', 'SELL', or 'HOLD'
            return 'HOLD'
        
        result = backtester.run(my_strategy)
        result.print_report()
    """
    
    def __init__(self, initial_capital: float = 10000, 
                 position_size_pct: float = 0.1,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.06):
        
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct  # 10% per trade
        self.stop_loss_pct = stop_loss_pct  # 3% stop loss
        self.take_profit_pct = take_profit_pct  # 6% take profit
        
        self.data: pd.DataFrame = None
        self.result = BacktestResult()
        
    def load_data(self, data_path: str, symbol: str = None) -> bool:
        """Load historical data from CSV or directory"""
        try:
            if os.path.isfile(data_path):
                self.data = pd.read_csv(data_path)
            elif os.path.isdir(data_path):
                # Load all CSVs and concatenate
                all_data = []
                files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
                
                if symbol:
                    files = [f for f in files if symbol in f]
                
                for f in sorted(files)[:10]:  # Limit to 10 files
                    df = pd.read_csv(os.path.join(data_path, f))
                    all_data.append(df)
                
                if all_data:
                    self.data = pd.concat(all_data, ignore_index=True)
            
            if self.data is not None and len(self.data) > 0:
                # Ensure we have required columns
                required = ['open', 'high', 'low', 'close', 'volume']
                for col in required:
                    if col not in self.data.columns:
                        print(f"Missing column: {col}")
                        return False
                
                # Sort by time if available
                if 'open_time' in self.data.columns:
                    self.data = self.data.sort_values('open_time').reset_index(drop=True)
                
                print(f"✅ Loaded {len(self.data)} candles")
                return True
            
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def load_dataframe(self, df: pd.DataFrame):
        """Load data from existing DataFrame"""
        self.data = df.copy()
        print(f"✅ Loaded {len(self.data)} candles from DataFrame")
    
    def run(self, signal_func, 
            lookback: int = 50,
            max_trades: int = None) -> BacktestResult:
        """
        Run backtest with given signal function
        
        Args:
            signal_func: Function that takes (df_slice, index) and returns 'BUY', 'SELL', or 'HOLD'
            lookback: Number of candles to look back for indicators
            max_trades: Maximum number of trades (None = unlimited)
        
        Returns:
            BacktestResult with all metrics
        """
        if self.data is None or len(self.data) < lookback:
            print("❌ Not enough data for backtest")
            return self.result
        
        self.result = BacktestResult()
        current_position: Optional[Trade] = None
        capital = self.initial_capital
        trade_count = 0
        
        print(f"🚀 Running backtest on {len(self.data)} candles...")
        
        for i in range(lookback, len(self.data)):
            # Get slice of data up to current candle
            df_slice = self.data.iloc[i-lookback:i+1].copy()
            current_candle = self.data.iloc[i]
            
            current_price = current_candle['close']
            current_time = current_candle.get('open_time', i)
            
            # Check if we have an open position
            if current_position is not None:
                # Check stop loss / take profit
                entry = current_position.entry_price
                
                if current_position.direction == 'LONG':
                    pnl_pct = (current_price - entry) / entry
                    
                    # Stop loss hit
                    if pnl_pct <= -self.stop_loss_pct:
                        current_position.close(current_time, entry * (1 - self.stop_loss_pct))
                        self.result.trades.append(current_position)
                        capital += current_position.pnl
                        current_position = None
                        continue
                    
                    # Take profit hit
                    if pnl_pct >= self.take_profit_pct:
                        current_position.close(current_time, entry * (1 + self.take_profit_pct))
                        self.result.trades.append(current_position)
                        capital += current_position.pnl
                        current_position = None
                        continue
                
                else:  # SHORT
                    pnl_pct = (entry - current_price) / entry
                    
                    if pnl_pct <= -self.stop_loss_pct:
                        current_position.close(current_time, entry * (1 + self.stop_loss_pct))
                        self.result.trades.append(current_position)
                        capital += current_position.pnl
                        current_position = None
                        continue
                    
                    if pnl_pct >= self.take_profit_pct:
                        current_position.close(current_time, entry * (1 - self.take_profit_pct))
                        self.result.trades.append(current_position)
                        capital += current_position.pnl
                        current_position = None
                        continue
            
            # Get signal from strategy
            try:
                signal = signal_func(df_slice, i)
            except Exception as e:
                signal = 'HOLD'
            
            # Execute signal
            if signal == 'BUY' and current_position is None:
                if max_trades and trade_count >= max_trades:
                    continue
                    
                size = capital * self.position_size_pct
                current_position = Trade(current_time, current_price, 'LONG', size)
                trade_count += 1
                
            elif signal == 'SELL' and current_position is None:
                if max_trades and trade_count >= max_trades:
                    continue
                    
                size = capital * self.position_size_pct
                current_position = Trade(current_time, current_price, 'SHORT', size)
                trade_count += 1
                
            elif signal == 'SELL' and current_position is not None and current_position.direction == 'LONG':
                # Close long position
                current_position.close(current_time, current_price)
                self.result.trades.append(current_position)
                capital += current_position.pnl
                current_position = None
                
            elif signal == 'BUY' and current_position is not None and current_position.direction == 'SHORT':
                # Close short position
                current_position.close(current_time, current_price)
                self.result.trades.append(current_position)
                capital += current_position.pnl
                current_position = None
        
        # Close any remaining position
        if current_position is not None:
            current_position.close(
                self.data.iloc[-1].get('open_time', len(self.data)),
                self.data.iloc[-1]['close']
            )
            self.result.trades.append(current_position)
        
        # Calculate metrics
        self.result.calculate_metrics(self.initial_capital)
        
        print(f"✅ Backtest complete: {len(self.result.trades)} trades")
        return self.result
    
    def run_with_model(self, model, lookback: int = 50) -> BacktestResult:
        """
        Run backtest using AIBrain model
        
        Args:
            model: MotherBrain instance with predict() method
            lookback: Number of candles for context
        """
        def model_signal(df, i):
            # Prepare market data
            market_data = {
                'klines': df,
                'df': df,
                'symbol': 'BACKTEST'
            }
            
            # Get prediction (0=BUY, 1=HOLD, 2=SELL)
            try:
                import asyncio
                signals = asyncio.run(model.collect_signals(market_data))
                # Use model's decision logic
                context = [
                    df.iloc[-1]['close'] / df.iloc[-10]['close'] - 1,  # Price change
                    0.5,  # Placeholder
                ]
                # Return based on model output
                return 'HOLD'
            except:
                return 'HOLD'
        
        return self.run(model_signal, lookback)
    
    def save_results(self, path: str = "R:/Redline_Data/backtest_results"):
        """Save backtest results to JSON"""
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(path, f"backtest_{timestamp}.json")
        
        results = {
            'timestamp': timestamp,
            'config': {
                'initial_capital': self.initial_capital,
                'position_size_pct': self.position_size_pct,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            },
            'metrics': self.result.to_dict(),
            'trades': [t.to_dict() for t in self.result.trades]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")


# === EXAMPLE STRATEGIES ===

def simple_sma_crossover(df: pd.DataFrame, i: int) -> str:
    """Simple SMA crossover strategy"""
    if len(df) < 50:
        return 'HOLD'
    
    sma_fast = df['close'].rolling(10).mean().iloc[-1]
    sma_slow = df['close'].rolling(30).mean().iloc[-1]
    
    sma_fast_prev = df['close'].rolling(10).mean().iloc[-2]
    sma_slow_prev = df['close'].rolling(30).mean().iloc[-2]
    
    # Golden cross
    if sma_fast > sma_slow and sma_fast_prev <= sma_slow_prev:
        return 'BUY'
    # Death cross
    elif sma_fast < sma_slow and sma_fast_prev >= sma_slow_prev:
        return 'SELL'
    
    return 'HOLD'


def rsi_oversold_overbought(df: pd.DataFrame, i: int) -> str:
    """RSI oversold/overbought strategy"""
    import pandas_ta as ta
    
    if len(df) < 20:
        return 'HOLD'
    
    rsi = df.ta.rsi(length=14).iloc[-1]
    
    if rsi < 30:
        return 'BUY'
    elif rsi > 70:
        return 'SELL'
    
    return 'HOLD'


# === QUICK TEST ===
if __name__ == "__main__":
    print("🧪 Running Backtester Test...")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    prices = [100]
    for _ in range(n-1):
        change = np.random.randn() * 0.02
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * n
    })
    
    backtester = Backtester(initial_capital=10000)
    backtester.load_dataframe(df)
    
    result = backtester.run(simple_sma_crossover)
    result.print_report()
