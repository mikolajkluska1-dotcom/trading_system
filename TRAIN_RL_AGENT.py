"""
AIBrain v5.0 - RL (PPO/DQN) Training Script
============================================
Trenuje agentów Reinforcement Learning do tradingu
"""
import os
import sys
import time
import random
from datetime import datetime

import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.AIBrain.ml.rl_agent import (
    TradingEnvironment,
    DQNAgent,
    PPOAgent,
    train_dqn,
    train_ppo
)

# =====================================================================
# CONFIGURATION
# =====================================================================


from agents.AIBrain.config import DATA_DIR, MODELS_DIR, RL_EPISODES, RL_LOOKBACK, RL_MAX_STEPS

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    # Data paths (will search all)
    'DATA_PATHS': [
        str(DATA_DIR / "1h"),
        str(DATA_DIR / "synthetic"),
        str(DATA_DIR / "4h"),
    ],
    'DQN_MODEL_PATH': str(MODELS_DIR / "dqn_agent.pth"),
    'PPO_MODEL_PATH': str(MODELS_DIR / "ppo_agent.pth"),
    
    # Training params
    'EPISODES': RL_EPISODES,
    'MAX_FILES': 30,
    'LOOKBACK': RL_LOOKBACK,
    'MAX_STEPS': RL_MAX_STEPS,
    
    # Environment
    'INITIAL_CAPITAL': 10000,
    'POSITION_SIZE': 0.2,  # 20% per trade
    'STOP_LOSS': 0.03,     # 3%
    'TAKE_PROFIT': 0.06,   # 6%
    
    # Algorithm selection
    'ALGORITHM': 'PPO',  # 'DQN' or 'PPO'
}


# =====================================================================
# DATA LOADING
# =====================================================================

def load_training_data(data_paths: list = None, max_files: int = 30) -> pd.DataFrame:
    """Load and concatenate kline data from multiple directories"""
    all_dfs = []
    
    if data_paths is None:
        data_paths = CONFIG['DATA_PATHS']
    
    count = 0
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"⚠️ Path not found (skipping): {data_path}")
            continue
            
        files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        files = sorted(files)
        
        print(f"📂 Found {len(files)} files in {data_path}")
        
        for f in files:
            try:
                df = pd.read_csv(os.path.join(data_path, f))
                if len(df) > CONFIG['LOOKBACK'] + 10:
                    all_dfs.append(df)
                    count += 1
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")
            
            if count >= max_files:
                break
        
        if count >= max_files:
            break
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Loaded {len(combined):,} candles total from {count} files")
        return combined
    
    return None


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_rl_agent(algorithm: str = 'PPO'):
    """Train RL agent (DQN or PPO)"""
    print("\n" + "="*60)
    print(f"🤖 AIBrain v5.0 - {algorithm} Training")
    print("="*60)
    
    # Load data
    # Use DATA_DIR from config (converted to string)
    df = load_training_data(str(CONFIG['DATA_DIR']), CONFIG['MAX_FILES'])
    if df is None:
        print("❌ No data loaded!")
        return None
    
    # Create environment
    env = TradingEnvironment(
        df=df,
        initial_capital=CONFIG['INITIAL_CAPITAL'],
        position_size_pct=CONFIG['POSITION_SIZE'],
        transaction_cost=0.001,
        lookback=CONFIG['LOOKBACK'],
        max_steps=CONFIG['MAX_STEPS']
    )
    
    print(f"\n🎮 Environment created:")
    print(f"   State dim:    {env.state_dim}")
    print(f"   Action dim:   {env.action_dim}")
    print(f"   Max steps:    {env.max_steps}")
    
    # Initialize agent
    if algorithm.upper() == 'DQN':
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=128,
            lr=0.0003,
            gamma=0.99
        )
        save_path = CONFIG['DQN_MODEL_PATH']
    else:  # PPO
        agent = PPOAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=128,
            lr=0.0003,
            gamma=0.99
        )
        save_path = CONFIG['PPO_MODEL_PATH']
    
    # Training loop
    best_return = -float('inf')
    returns_history = []
    start_time = time.time()
    
    print(f"\n🚀 Starting {algorithm} training for {CONFIG['EPISODES']} episodes...")
    
    for episode in range(CONFIG['EPISODES']):
        metrics = agent.train_episode(env)
        returns_history.append(metrics['total_return'])
        
        # Track best
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            agent.save(save_path)
            print(f"\n🎯 New best return: {best_return:.2f}%")
        
        # Log progress
        if episode % 10 == 0:
            avg_return = np.mean(returns_history[-50:]) if len(returns_history) >= 50 else np.mean(returns_history)
            
            extra_info = ""
            if algorithm.upper() == 'DQN':
                extra_info = f"Eps={agent.epsilon:.3f}"
            
            print(f"Episode {episode:4d}: "
                  f"Return={metrics['total_return']:6.2f}% | "
                  f"Trades={metrics['num_trades']:3d} | "
                  f"Wins={metrics['winning_trades']:2d} | "
                  f"Avg50={avg_return:.2f}% | "
                  f"{extra_info}")
    
    # Training summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("📊 TRAINING COMPLETE")
    print("="*60)
    print(f"Algorithm:       {algorithm}")
    print(f"Episodes:        {CONFIG['EPISODES']}")
    print(f"Best Return:     {best_return:.2f}%")
    print(f"Final Avg (50):  {np.mean(returns_history[-50:]):.2f}%")
    print(f"Training Time:   {elapsed/60:.1f} minutes")
    print(f"Model saved:     {save_path}")
    print("="*60)
    
    return agent


def evaluate_agent(agent, algorithm: str):
    """Evaluate trained agent on test data"""
    print("\n🧪 Evaluating agent...")
    
    # Load different data for evaluation
    df = load_training_data(CONFIG['DATA_PATH'], 5)  # Use fewer files
    if df is None:
        return
    
    env = TradingEnvironment(
        df=df,
        initial_capital=CONFIG['INITIAL_CAPITAL'],
        position_size_pct=CONFIG['POSITION_SIZE'],
        lookback=CONFIG['LOOKBACK'],
        max_steps=1000  # Longer evaluation
    )
    
    # Run evaluation (no training)
    state = env.reset()
    total_reward = 0
    
    while not env.done:
        if algorithm.upper() == 'DQN':
            action = agent.select_action(state, training=False)
        else:
            action, _, _ = agent.select_action(state, training=False)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
    
    metrics = env.get_metrics()
    
    print("\n📈 Evaluation Results:")
    print(f"   Total Return:   {metrics['total_return']:.2f}%")
    print(f"   Num Trades:     {metrics['num_trades']}")
    print(f"   Win Rate:       {metrics['winning_trades']/max(metrics['num_trades'],1)*100:.1f}%")
    print(f"   Max Drawdown:   {metrics['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")


def compare_algorithms():
    """Train and compare both DQN and PPO"""
    print("\n" + "="*60)
    print("🏁 ALGORITHM COMPARISON: DQN vs PPO")
    print("="*60)
    
    results = {}
    
    # Train DQN
    print("\n>>> Training DQN...")
    dqn_agent = train_rl_agent('DQN')
    if dqn_agent:
        results['DQN'] = {
            'episodes': dqn_agent.episodes,
        }
    
    # Train PPO
    print("\n>>> Training PPO...")
    ppo_agent = train_rl_agent('PPO')
    if ppo_agent:
        results['PPO'] = {
            'episodes': ppo_agent.episodes,
        }
    
    print("\n" + "="*60)
    print("🏆 COMPARISON RESULTS")
    print("="*60)
    for algo, res in results.items():
        print(f"{algo}: Episodes={res['episodes']}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL Agent for Trading')
    parser.add_argument('--algo', type=str, default='PPO', choices=['DQN', 'PPO', 'BOTH'],
                        help='Algorithm to train')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    CONFIG['EPISODES'] = args.episodes
    CONFIG['ALGORITHM'] = args.algo
    
    try:
        if args.algo == 'BOTH':
            compare_algorithms()
        else:
            agent = train_rl_agent(args.algo)
            
            if args.evaluate and agent:
                evaluate_agent(agent, args.algo)
                
    except KeyboardInterrupt:
        print("\n\n👋 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
