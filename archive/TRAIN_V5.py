# TRAIN_V5.py - COMPLETE AI TRAINING WITH ALL DATA
# Uses ALL 12 data sources for maximum AI intelligence

import os
import sys
import time
import signal
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Quiet mode
logging.getLogger('MOTHER_BRAIN').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIG
# =============================================================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVALS = ["1m", "5m", "1h", "1d"]
EPOCHS = 50
STEPS_PER_STATUS = 50000
SAVE_EVERY_STEPS = 100000

# Reward
MIN_CONFIDENCE = 0.78
WIN_MULTIPLIER = 15.0
LOSS_MULTIPLIER = 0.3
HOLD_REWARD = 0.02
FEE = 0.01

# Paths
BULK_PATH = "R:/Redline_Data/bulk_data"
DATA_PATH = "R:/Redline_Data"
CHECKPOINT_PATH = "C:/Users/Mikołaj/trading_system/models/checkpoints"
AI_MODELS_PATH = "C:/Users/Mikołaj/trading_system/models/ai_logic"

STOP_TRAINING = False

def signal_handler(sig, frame):
    global STOP_TRAINING
    print("\n🛑 STOPPING... Saving models...")
    STOP_TRAINING = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# ALL DATA LOADER (12 sources)
# =============================================================================

class AllDataLoader:
    def __init__(self):
        print("\n📊 Loading ALL data sources...")
        
        self.fear_greed = None
        self.global_market = None
        self.trending = None
        self.funding_rates = {}
        self.open_interest = {}
        self.long_short = {}
        self.taker_ratio = {}
        self.top_traders = {}
        self.order_book = {}
        self.whale_trades = None
        self.exchange_stats = {}
        self.volatility = {}
        
        self._load_all()
        
    def _load_all(self):
        # 1. Fear & Greed
        path = os.path.join(DATA_PATH, "sentiment", "fear_greed.csv")
        if os.path.exists(path):
            self.fear_greed = pd.read_csv(path)
            print(f"   ✅ Fear & Greed: {len(self.fear_greed)} days")
        
        # 2. Global Market
        path = os.path.join(DATA_PATH, "market", "global_market.csv")
        if os.path.exists(path):
            self.global_market = pd.read_csv(path)
            print(f"   ✅ Global Market: loaded")
        
        # 3. Trending
        path = os.path.join(DATA_PATH, "sentiment", "trending_coins.csv")
        if os.path.exists(path):
            self.trending = pd.read_csv(path)
            print(f"   ✅ Trending: {len(self.trending)} coins")
        
        # 4. Funding Rates
        path = os.path.join(DATA_PATH, "futures", "funding_rates.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.funding_rates[s] = df[df['symbol'] == s]
            print(f"   ✅ Funding Rates: {len(df)} records")
        
        # 5. Open Interest
        path = os.path.join(DATA_PATH, "futures", "open_interest.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.open_interest[s] = df[df['symbol'] == s]
            print(f"   ✅ Open Interest: {len(df)} records")
        
        # 6. Long/Short Ratio
        path = os.path.join(DATA_PATH, "futures", "long_short_ratio.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.long_short[s] = df[df['symbol'] == s]
            print(f"   ✅ Long/Short: {len(df)} records")
        
        # 7. Taker Ratio
        path = os.path.join(DATA_PATH, "futures", "taker_ratio.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.taker_ratio[s] = df[df['symbol'] == s]
            print(f"   ✅ Taker Ratio: {len(df)} records")
        
        # 8. Top Traders
        path = os.path.join(DATA_PATH, "futures", "top_trader_positions.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                self.top_traders[s] = df[df['symbol'] == s]
            print(f"   ✅ Top Traders: {len(df)} records")
        
        # 9. Order Book
        path = os.path.join(DATA_PATH, "orderbook", "depth_snapshot.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                rows = df[df['symbol'] == s]
                if len(rows) > 0:
                    self.order_book[s] = rows.iloc[-1].to_dict()
            print(f"   ✅ Order Book: {len(df)} snapshots")
        
        # 10. Whale Trades
        path = os.path.join(DATA_PATH, "whales", "large_trades.csv")
        if os.path.exists(path):
            self.whale_trades = pd.read_csv(path)
            print(f"   ✅ Whale Trades: {len(self.whale_trades)} trades")
        
        # 11. Exchange Stats
        path = os.path.join(DATA_PATH, "exchange", "24h_stats.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                rows = df[df['symbol'] == s]
                if len(rows) > 0:
                    self.exchange_stats[s] = rows.iloc[-1].to_dict()
            print(f"   ✅ Exchange Stats: {len(df)} records")
        
        # 12. Volatility
        path = os.path.join(DATA_PATH, "analytics", "volatility.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            for s in SYMBOLS:
                rows = df[df['symbol'] == s]
                if len(rows) > 0:
                    self.volatility[s] = rows.iloc[-1].to_dict()
            print(f"   ✅ Volatility: {len(df)} records")
    
    # === GETTERS ===
    
    def get_fear_greed(self):
        if self.fear_greed is None: return 50
        try: return int(self.fear_greed.iloc[0]['value'])
        except: return 50
    
    def get_btc_dominance(self):
        if self.global_market is None: return 50
        try: return float(self.global_market.iloc[0]['btc_dominance'])
        except: return 50
    
    def get_market_cap_change(self):
        if self.global_market is None: return 0
        try: return float(self.global_market.iloc[0]['market_cap_change_24h'])
        except: return 0
    
    def get_funding_rate(self, symbol):
        if symbol not in self.funding_rates: return 0.0
        try: return float(self.funding_rates[symbol].iloc[-1]['fundingRate'])
        except: return 0.0
    
    def get_long_short_ratio(self, symbol):
        if symbol not in self.long_short: return 1.0
        try: return float(self.long_short[symbol].iloc[-1]['longShortRatio'])
        except: return 1.0
    
    def get_taker_ratio(self, symbol):
        if symbol not in self.taker_ratio: return 1.0
        try: return float(self.taker_ratio[symbol].iloc[-1]['buySellRatio'])
        except: return 1.0
    
    def get_top_trader_ratio(self, symbol):
        if symbol not in self.top_traders: return 1.0
        try: return float(self.top_traders[symbol].iloc[-1]['longShortRatio'])
        except: return 1.0
    
    def get_order_book_ratio(self, symbol):
        if symbol not in self.order_book: return 1.0
        try: return float(self.order_book[symbol].get('bid_ask_ratio', 1.0))
        except: return 1.0
    
    def get_whale_bias(self, symbol):
        if self.whale_trades is None: return 0.0
        try:
            trades = self.whale_trades[self.whale_trades['symbol'] == symbol]
            if len(trades) == 0: return 0.0
            buys = len(trades[trades['is_buyer_maker'] == False])
            sells = len(trades[trades['is_buyer_maker'] == True])
            total = buys + sells
            return (buys - sells) / total if total > 0 else 0.0
        except: return 0.0
    
    def get_volatility(self, symbol):
        if symbol not in self.volatility: return 50.0
        try: return float(self.volatility[symbol].get('volatility_30d', 50.0))
        except: return 50.0
    
    def get_volume_24h(self, symbol):
        if symbol not in self.exchange_stats: return 0
        try: return float(self.exchange_stats[symbol].get('quote_volume_24h', 0))
        except: return 0

# =============================================================================
# AI TEAM
# =============================================================================

class AITeam:
    def __init__(self, data_loader):
        print("\n🧠 Initializing AI Team...")
        
        from agents.AIBrain.ml.mother_brain import MotherBrain
        
        self.mother = MotherBrain()
        self.mother.current_balance = 10000.0
        self.mother.total_trades = 0
        self.mother.profitable_trades = 0
        self.mother.total_profit = 0.0
        self.mother.generation = 1
        
        self.data = data_loader
        
        self.wins = 0
        self.losses = 0
        self.holds = 0
        self.total_steps = 0
        self.start_time = datetime.now()
        
        print(f"   ✅ Mother Brain ready (fresh)")
        print(f"   ✅ {len(self.mother.children)} child agents")
        print(f"   ✅ 12 data sources integrated")
    
    def get_decision(self, row, symbol, scenarios):
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        conf = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        # Technical signals
        for s in scenarios:
            votes[s['signal']] += 1
            conf[s['signal']].append(s['confidence'])
        
        # === ALL 12 DATA SOURCES ===
        
        # 1. Fear & Greed
        fg = self.data.get_fear_greed()
        if fg < 20:  # Extreme fear = strong BUY
            votes['BUY'] += 3
            conf['BUY'].append(0.85)
        elif fg < 35:  # Fear = BUY
            votes['BUY'] += 2
            conf['BUY'].append(0.7)
        elif fg > 80:  # Extreme greed = strong SELL
            votes['SELL'] += 3
            conf['SELL'].append(0.85)
        elif fg > 65:  # Greed = SELL
            votes['SELL'] += 2
            conf['SELL'].append(0.7)
        
        # 2. BTC Dominance (if rising + BTC = BUY, if falling = ALTs)
        btc_dom = self.data.get_btc_dominance()
        if symbol == 'BTCUSDT' and btc_dom > 55:
            votes['BUY'] += 1
            conf['BUY'].append(0.6)
        
        # 3. Market Cap Change
        mc_change = self.data.get_market_cap_change()
        if mc_change > 3:
            votes['BUY'] += 1
            conf['BUY'].append(0.55)
        elif mc_change < -3:
            votes['SELL'] += 1
            conf['SELL'].append(0.55)
        
        # 4. Funding Rate (contrarian)
        fr = self.data.get_funding_rate(symbol)
        if fr > 0.001:  # High = overleveraged longs
            votes['SELL'] += 2
            conf['SELL'].append(0.7)
        elif fr < -0.001:  # Negative = overleveraged shorts
            votes['BUY'] += 2
            conf['BUY'].append(0.7)
        
        # 5. Long/Short Ratio (contrarian)
        ls = self.data.get_long_short_ratio(symbol)
        if ls > 2.5:
            votes['SELL'] += 2
            conf['SELL'].append(0.65)
        elif ls < 0.5:
            votes['BUY'] += 2
            conf['BUY'].append(0.65)
        
        # 6. Taker Ratio (aggressive trading)
        tr = self.data.get_taker_ratio(symbol)
        if tr > 1.3:  # Aggressive buyers
            votes['BUY'] += 1
            conf['BUY'].append(0.6)
        elif tr < 0.7:  # Aggressive sellers
            votes['SELL'] += 1
            conf['SELL'].append(0.6)
        
        # 7. Top Traders
        tt = self.data.get_top_trader_ratio(symbol)
        if tt > 1.5:
            votes['BUY'] += 2
            conf['BUY'].append(0.75)
        elif tt < 0.6:
            votes['SELL'] += 2
            conf['SELL'].append(0.75)
        
        # 8. Order Book
        ob = self.data.get_order_book_ratio(symbol)
        if ob > 1.5:  # Strong bid support
            votes['BUY'] += 1
            conf['BUY'].append(0.6)
        elif ob < 0.6:  # Strong ask pressure
            votes['SELL'] += 1
            conf['SELL'].append(0.6)
        
        # 9. Whale Trades
        wb = self.data.get_whale_bias(symbol)
        if wb > 0.4:
            votes['BUY'] += 3
            conf['BUY'].append(0.8)
        elif wb < -0.4:
            votes['SELL'] += 3
            conf['SELL'].append(0.8)
        
        # 10. Volatility (adjust confidence)
        vol = self.data.get_volatility(symbol)
        vol_factor = 1.0
        if vol > 60:  # High vol = less confident
            vol_factor = 0.85
        elif vol < 30:  # Low vol = more confident
            vol_factor = 1.1
        
        # === FINAL DECISION ===
        best = max(votes, key=votes.get)
        
        if conf[best]:
            avg_conf = sum(conf[best]) / len(conf[best]) * vol_factor
        else:
            avg_conf = 0.5
        
        # Require consensus
        if avg_conf < MIN_CONFIDENCE or votes[best] < 4:
            return 'HOLD', 0.9, 0
        
        return best, min(avg_conf, 0.99), votes[best]
    
    def calculate_reward(self, action, confidence, price_change, confirmations):
        if action == 'HOLD':
            self.holds += 1
            reward = HOLD_REWARD
            if abs(price_change) > 1.5:
                reward += 0.02  # Bonus for avoiding volatile moves
            return reward
        
        if action == 'BUY':
            pnl = price_change - FEE
        else:
            pnl = -price_change - FEE
        
        if pnl > 0:
            self.wins += 1
            reward = pnl * WIN_MULTIPLIER * (1 + confidence) * (1 + confirmations * 0.1)
        else:
            self.losses += 1
            reward = pnl * LOSS_MULTIPLIER
        
        return reward
    
    def train_step(self, row, next_row, symbol, scenarios):
        self.total_steps += 1
        
        action, confidence, confirmations = self.get_decision(row, symbol, scenarios)
        price_change = (next_row['close'] - row['close']) / row['close'] * 100
        reward = self.calculate_reward(action, confidence, price_change, confirmations)
        
        self.mother.current_balance += reward
        if reward > 0:
            self.mother.profitable_trades += 1
        self.mother.total_trades += 1
        
        if self.mother.total_trades % 100 == 0:
            self.mother.generation += 1
        
        return reward
    
    def get_win_rate(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0
    
    def save_models(self):
        os.makedirs(AI_MODELS_PATH, exist_ok=True)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        
        state = {
            'generation': self.mother.generation,
            'balance': self.mother.current_balance,
            'total_trades': self.mother.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'holds': self.holds,
            'win_rate': self.get_win_rate(),
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat(),
            'version': 'V5_12_data_sources'
        }
        
        with open(os.path.join(AI_MODELS_PATH, 'v5_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        self.mother.save_checkpoint()
        print(f"   💾 Models saved")

# =============================================================================
# DATA LOADER (Klines)
# =============================================================================

class KlinesLoader:
    @staticmethod
    def load_all():
        all_data = {}
        total = 0
        
        print("\n📊 Loading klines...")
        
        for interval in INTERVALS:
            for symbol in SYMBOLS:
                path = os.path.join(BULK_PATH, "klines", interval, symbol)
                if os.path.exists(path):
                    dfs = []
                    for f in sorted(os.listdir(path)):
                        if f.endswith('.csv'):
                            try:
                                df = pd.read_csv(os.path.join(path, f), header=None)
                                if len(df.columns) >= 6:
                                    df = df.iloc[:, :6]
                                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                    dfs.append(df)
                            except:
                                pass
                    
                    if dfs:
                        combined = pd.concat(dfs, ignore_index=True)
                        combined = combined.drop_duplicates('timestamp').sort_values('timestamp')
                        
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            combined[col] = pd.to_numeric(combined[col], errors='coerce')
                        
                        combined = KlinesLoader.add_indicators(combined)
                        combined = combined.dropna().reset_index(drop=True)
                        
                        if len(combined) > 100:
                            key = f"{symbol}_{interval}"
                            all_data[key] = combined
                            total += len(combined)
                            print(f"   ✅ {key}: {len(combined):,}")
        
        print(f"\n   📈 Total: {len(all_data)} datasets, {total:,} candles")
        return all_data
    
    @staticmethod
    def add_indicators(df):
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))
        
        df['bb_mid'] = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * std
        df['bb_lower'] = df['bb_mid'] - 2 * std
        
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        return df

# =============================================================================
# SCENARIO GENERATOR
# =============================================================================

class Scenarios:
    @staticmethod
    def generate(row):
        scenarios = []
        
        rsi = row.get('rsi', 50)
        ema_9 = row.get('ema_9', row['close'])
        ema_21 = row.get('ema_21', row['close'])
        close = row['close']
        bb_upper = row.get('bb_upper', close * 1.02)
        bb_lower = row.get('bb_lower', close * 0.98)
        macd = row.get('macd', 0)
        macd_sig = row.get('macd_signal', 0)
        
        # RSI
        if rsi < 25: scenarios.append({'signal': 'BUY', 'confidence': 0.85})
        elif rsi < 35: scenarios.append({'signal': 'BUY', 'confidence': 0.7})
        elif rsi > 75: scenarios.append({'signal': 'SELL', 'confidence': 0.85})
        elif rsi > 65: scenarios.append({'signal': 'SELL', 'confidence': 0.7})
        
        # EMA
        if ema_9 > ema_21 * 1.002: scenarios.append({'signal': 'BUY', 'confidence': 0.75})
        elif ema_9 < ema_21 * 0.998: scenarios.append({'signal': 'SELL', 'confidence': 0.75})
        
        # Bollinger
        if close <= bb_lower: scenarios.append({'signal': 'BUY', 'confidence': 0.8})
        elif close >= bb_upper: scenarios.append({'signal': 'SELL', 'confidence': 0.8})
        
        # MACD
        if macd > macd_sig and macd < 0: scenarios.append({'signal': 'BUY', 'confidence': 0.82})
        elif macd < macd_sig and macd > 0: scenarios.append({'signal': 'SELL', 'confidence': 0.82})
        
        if not scenarios:
            scenarios.append({'signal': 'HOLD', 'confidence': 0.9})
        
        return scenarios

# =============================================================================
# TRAINER
# =============================================================================

class TrainerV5:
    def __init__(self):
        self.data_loader = AllDataLoader()
        self.team = AITeam(self.data_loader)
        self.klines = KlinesLoader.load_all()
    
    def train(self):
        global STOP_TRAINING
        
        if not self.klines:
            print("❌ No data!")
            return
        
        print("\n" + "=" * 60)
        print("🚀 TRAINING V5 - 12 DATA SOURCES")
        print("=" * 60)
        print(f"   📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📊 Datasets: {len(self.klines)}")
        print(f"   🔄 Epochs: {EPOCHS}")
        print("=" * 60)
        
        for epoch in range(1, EPOCHS + 1):
            if STOP_TRAINING: break
            
            epoch_start = time.time()
            epoch_steps = 0
            
            print(f"\n📚 EPOCH {epoch}/{EPOCHS}")
            
            for key, df in self.klines.items():
                if STOP_TRAINING: break
                
                symbol = key.split('_')[0]
                
                for i in range(60, len(df) - 1):
                    if STOP_TRAINING: break
                    
                    row = df.iloc[i]
                    next_row = df.iloc[i + 1]
                    
                    scenarios = Scenarios.generate(row)
                    self.team.train_step(row, next_row, symbol, scenarios)
                    epoch_steps += 1
                    
                    if self.team.total_steps % STEPS_PER_STATUS == 0:
                        self.print_status()
                    
                    if self.team.total_steps % SAVE_EVERY_STEPS == 0:
                        self.team.save_models()
            
            epoch_time = time.time() - epoch_start
            wr = self.team.get_win_rate() * 100
            print(f"\n   ⏱️  Epoch {epoch}: {epoch_time:.1f}s | Win: {wr:.1f}% | Balance: ${self.team.mother.current_balance:,.0f}")
        
        self.finish()
    
    def print_status(self):
        wr = self.team.get_win_rate() * 100
        elapsed = (datetime.now() - self.team.start_time).total_seconds()
        speed = self.team.total_steps / elapsed if elapsed > 0 else 0
        
        print(f"   📊 {self.team.total_steps:,} | Win: {wr:.1f}% | ${self.team.mother.current_balance:,.0f} | Gen: {self.team.mother.generation} | {speed:.0f}/s")
    
    def finish(self):
        print("\n" + "=" * 60)
        print("💾 SAVING...")
        self.team.save_models()
        
        wr = self.team.get_win_rate() * 100
        
        print("\n" + "=" * 60)
        print("✅ TRAINING V5 COMPLETE!")
        print("=" * 60)
        print(f"   📊 Steps: {self.team.total_steps:,}")
        print(f"   💰 Balance: ${self.team.mother.current_balance:,.2f}")
        print(f"   🎯 Win Rate: {wr:.1f}%")
        print(f"   ✅ Wins: {self.team.wins:,} | ❌ Losses: {self.team.losses:,} | ⏸️ Holds: {self.team.holds:,}")
        print(f"   🧬 Generation: {self.team.mother.generation}")
        print("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(AI_MODELS_PATH, exist_ok=True)
    
    TrainerV5().train()
