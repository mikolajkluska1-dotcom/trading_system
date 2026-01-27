import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from sqlalchemy import create_engine
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# --- KONFIGURACJA ---
# Używamy portu 5435, bo łączymy się z Windowsa do Dockera
DB_URL = "postgresql://redline_user:redline_pass@localhost:5435/redline_db"
SYMBOL = "BTC/USDT" # Trenujemy na Królu (najwięcej danych)
WINDOW_SIZE = 30    # AI widzi 30 ostatnich sekund

# --- 1. ŚRODOWISKO HANDLOWE (GYM) ---
class CryptoTradingEnv(gym.Env):
    def __init__(self, df):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.render_mode = None
        
        # Akcje: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        
        # Obserwacja: Ostatnie 30 cen (OHLCV)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW_SIZE, 5), dtype=np.float32
        )
        
        self.reset_vars()

    def reset_vars(self):
        self.current_step = WINDOW_SIZE
        self.balance = 1000.0     # Startowe USDT
        self.crypto_held = 0.0    # Startowe BTC
        self.net_worth = 1000.0
        self.max_net_worth = 1000.0
        self.trades = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_vars()
        return self._next_observation(), {}

    def _next_observation(self):
        # Pobieramy wycinek danych
        frame = self.df.iloc[self.current_step - WINDOW_SIZE : self.current_step]
        obs = frame[['open', 'high', 'low', 'close', 'volume']].values
        
        # Prosta normalizacja (dzielimy przez pierwszą cenę w oknie)
        base_price = obs[0, 3] # Close price z początku okna
        if base_price > 0:
            obs = obs / base_price
            
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth

        # --- LOGIKA HANDLU ---
        if action == 1: # BUY
            if self.balance > 10: # Minimum 10$
                amount_to_buy = self.balance / current_price
                self.crypto_held += amount_to_buy * 0.999 # Prowizja 0.1%
                self.balance = 0
                self.trades += 1
                
        elif action == 2: # SELL
            if self.crypto_held > 0:
                amount_sold = self.crypto_held * current_price
                self.balance += amount_sold * 0.999 # Prowizja 0.1%
                self.crypto_held = 0
                self.trades += 1

        # Przesuwamy czas
        self.current_step += 1
        
        # Aktualizacja wartości portfela
        self.net_worth = self.balance + (self.crypto_held * current_price)
        
        # --- NAGRODA / KARA (RL) ---
        reward = self.net_worth - prev_net_worth
        
        # Bonus za zysk, kara za stratę (wzmocnienie)
        if reward > 0:
            reward *= 1.5
        
        done = self.current_step >= len(self.df) - 1
        
        return self._next_observation(), reward, done, False, {}

# --- 2. POBIERANIE DANYCH ---
def get_data():
    print(f"📥 [1/4] Łączenie z bazą danych dla {SYMBOL}...")
    try:
        engine = create_engine(DB_URL)
        # Pobieramy max 500,000 rekordów żeby nie zapchać pamięci RAM na noc
        query = f"SELECT * FROM market_candles WHERE symbol = '{SYMBOL}' ORDER BY time ASC LIMIT 500000"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("❌ Błąd: Brak danych w bazie! Czy Snajper działał?")
            return None
            
        # Konwersja na float
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        print(f"✅ Pobrano {len(df)} rekordów. Przygotowanie do treningu...")
        return df
    except Exception as e:
        print(f"❌ Błąd połączenia z bazą: {e}")
        return None

# --- 3. START TRENINGU ---
if __name__ == "__main__":
    df = get_data()
    
    if df is not None:
        print("🏋️ [2/4] Tworzenie środowiska treningowego...")
        env = DummyVecEnv([lambda: CryptoTradingEnv(df)])
        
        print("🧠 [3/4] Inicjalizacja modelu PPO...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, batch_size=2048)
        
        print("\n🚀 [4/4] START TRENINGU NOCNEGO (Naciśnij Ctrl+C by przerwać wcześniej)")
        print("To potrwa kilka godzin. Idź spać. AI pracuje.\n")
        
        try:
            # Trenujemy na 1 milionie kroków (lub ile zdąży przez noc)
            model.learn(total_timesteps=1000000) 
        except KeyboardInterrupt:
            print("\n⚠️ Przerwano ręcznie. Zapisywanie tego co jest...")
        
        model.save("redline_brain_nightly")
        print("\n💾 Model zapisany jako 'redline_brain_nightly.zip'.")
        print("✅ DOBRANOC! JESTEŚMY GOTOWI NA RANO.")