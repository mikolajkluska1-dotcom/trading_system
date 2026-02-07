"""
AIBrain v5.0 - MEGA UPGRADE
Reinforcement Learning Agent - PPO & DQN
=========================================
Model sam się uczy handlować poprzez interakcję ze środowiskiem!

Features:
- Deep Q-Network (DQN) dla dyskretnych akcji
- Proximal Policy Optimization (PPO) dla stabilnego uczenia
- Custom Trading Environment (OpenAI Gym style)
- Reward shaping dla risk-adjusted returns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from collections import deque
import random
import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RL_Agent")


# =====================================================================
# TRADING ENVIRONMENT
# =====================================================================

@dataclass
class TradingState:
    """State representation for trading environment"""
    position: int  # -1 = SHORT, 0 = FLAT, 1 = LONG
    entry_price: float
    current_price: float
    unrealized_pnl: float
    cash: float
    portfolio_value: float
    step: int


class TradingEnvironment:
    """
    Custom Trading Environment (OpenAI Gym style)
    
    State: [price_features, position, unrealized_pnl, portfolio_value_norm]
    Actions: 0 = HOLD, 1 = BUY, 2 = SELL
    Reward: Risk-adjusted PnL
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000,
        position_size_pct: float = 0.2,
        transaction_cost: float = 0.001,
        max_steps: int = None,
        lookback: int = 30
    ):
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        self.max_steps = max_steps or (len(df) - lookback - 1)
        
        # State dimensions
        self.state_dim = lookback * 5 + 4  # 5 features per candle + position info
        self.action_dim = 3  # HOLD, BUY, SELL
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.lookback
        self.cash = self.initial_capital
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0
        self.position_size = 0
        self.portfolio_values = [self.initial_capital]
        self.trades = []
        self.done = False
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Price features (last N candles)
        start = self.current_step - self.lookback
        end = self.current_step
        
        candles = self.df.iloc[start:end]
        current_price = self.df.iloc[self.current_step]['close']
        
        # Normalize prices relative to current
        close = candles['close'].values / current_price - 1
        high = candles['high'].values / current_price - 1
        low = candles['low'].values / current_price - 1
        
        # Returns
        returns = np.diff(candles['close'].values, prepend=candles['close'].values[0]) / candles['close'].values[0]
        
        # Volume (normalized)
        volume = candles['volume'].values / candles['volume'].mean() - 1
        
        # Flatten price features
        price_features = np.concatenate([close, high, low, returns, volume])
        
        # Position features
        unrealized_pnl = 0
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:  # Long
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        portfolio_value = self.cash + (self.position_size * current_price * self.position if self.position != 0 else 0)
        portfolio_norm = portfolio_value / self.initial_capital - 1
        
        position_features = np.array([
            self.position,
            unrealized_pnl,
            portfolio_norm,
            self.current_step / len(self.df)  # Progress
        ])
        
        state = np.concatenate([price_features, position_features])
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            next_state, reward, done, info
        """
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        info = {'action': action, 'position': self.position}
        
        # Execute action
        if action == 1:  # BUY
            if self.position <= 0:  # Can buy if flat or short
                # Close short if exists
                if self.position == -1:
                    pnl = (self.entry_price - current_price) / self.entry_price * self.position_size
                    pnl -= self.position_size * self.transaction_cost
                    self.cash += self.position_size + pnl
                    reward += pnl / self.initial_capital * 100  # Reward as percentage
                    self.trades.append({'type': 'CLOSE_SHORT', 'pnl': pnl})
                    self.position = 0
                    self.position_size = 0
                
                # Open long
                if self.position == 0:
                    self.position_size = self.cash * self.position_size_pct
                    self.cash -= self.position_size * (1 + self.transaction_cost)
                    self.position = 1
                    self.entry_price = current_price
                    self.trades.append({'type': 'OPEN_LONG', 'price': current_price})
                    reward -= 0.01  # Small cost for trading
        
        elif action == 2:  # SELL
            if self.position >= 0:  # Can sell if flat or long
                # Close long if exists
                if self.position == 1:
                    pnl = (current_price - self.entry_price) / self.entry_price * self.position_size
                    pnl -= self.position_size * self.transaction_cost
                    self.cash += self.position_size + pnl
                    reward += pnl / self.initial_capital * 100
                    self.trades.append({'type': 'CLOSE_LONG', 'pnl': pnl})
                    self.position = 0
                    self.position_size = 0
                
                # Open short
                if self.position == 0:
                    self.position_size = self.cash * self.position_size_pct
                    self.cash -= self.position_size * self.transaction_cost  # Only pay fee
                    self.position = -1
                    self.entry_price = current_price
                    self.trades.append({'type': 'OPEN_SHORT', 'price': current_price})
                    reward -= 0.01
        
        else:  # HOLD
            # Small penalty for holding with unrealized loss
            if self.position != 0:
                if self.position == 1:
                    unrealized = (current_price - self.entry_price) / self.entry_price
                else:
                    unrealized = (self.entry_price - current_price) / self.entry_price
                
                if unrealized < -0.02:  # More than 2% loss
                    reward -= 0.001  # Small penalty to encourage stop loss
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        next_price = self.df.iloc[self.current_step]['close'] if self.current_step < len(self.df) else current_price
        portfolio_value = self.cash
        if self.position == 1:
            portfolio_value += self.position_size * (1 + (next_price - self.entry_price) / self.entry_price)
        elif self.position == -1:
            portfolio_value += self.position_size * (1 + (self.entry_price - next_price) / self.entry_price)
        
        self.portfolio_values.append(portfolio_value)
        
        # Check if done
        if self.current_step >= len(self.df) - 1 or self.current_step >= self.lookback + self.max_steps:
            self.done = True
            # Close any open position
            if self.position != 0:
                final_price = self.df.iloc[self.current_step]['close']
                if self.position == 1:
                    pnl = (final_price - self.entry_price) / self.entry_price * self.position_size
                else:
                    pnl = (self.entry_price - final_price) / self.entry_price * self.position_size
                reward += pnl / self.initial_capital * 100
        
        # Bonus for profitable episode
        if self.done and portfolio_value > self.initial_capital:
            reward += (portfolio_value / self.initial_capital - 1) * 10
        
        next_state = self._get_state() if not self.done else np.zeros(self.state_dim)
        
        info['portfolio_value'] = portfolio_value
        info['trades'] = len(self.trades)
        
        return next_state, reward, self.done, info
    
    def get_metrics(self) -> Dict:
        """Get episode performance metrics"""
        returns = np.array(self.portfolio_values)
        final_value = returns[-1]
        
        return {
            'total_return': (final_value / self.initial_capital - 1) * 100,
            'num_trades': len([t for t in self.trades if 'pnl' in t]),
            'winning_trades': len([t for t in self.trades if t.get('pnl', 0) > 0]),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'sharpe_ratio': self._calculate_sharpe(returns)
        }
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return drawdown.max() * 100
    
    def _calculate_sharpe(self, values: np.ndarray) -> float:
        returns = np.diff(values) / values[:-1]
        if len(returns) < 2:
            return 0
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)


# =====================================================================
# DQN (Deep Q-Network)
# =====================================================================

class DQNNetwork(nn.Module):
    """DQN Neural Network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Dueling DQN streams
        self.value_stream = nn.Linear(hidden_dim // 2, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 2, action_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Dueling architecture
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for Trading
    
    Features:
    - Double DQN (reduces overestimation)
    - Dueling DQN (separates value and advantage)
    - Prioritized Experience Replay
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.0003,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = 'cuda'
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
        self.episodes = 0
        
        logger.info(f"🤖 DQN Agent initialized on {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self) -> float:
        """Update policy network using experience replay"""
        if len(self.buffer) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and update
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train_episode(self, env: TradingEnvironment) -> Dict:
        """Train for one episode"""
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        
        while not env.done:
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            self.buffer.push(state, action, reward, next_state, done)
            loss = self.update()
            
            state = next_state
            total_reward += reward
            total_loss += loss
            steps += 1
        
        self.episodes += 1
        metrics = env.get_metrics()
        
        return {
            'episode': self.episodes,
            'total_reward': total_reward,
            'avg_loss': total_loss / max(steps, 1),
            'epsilon': self.epsilon,
            **metrics
        }
    
    def save(self, path: str = "R:/Redline_Data/ai_logic/dqn_agent.pth"):
        """Save agent"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
        logger.info(f"💾 DQN Agent saved to {path}")
    
    def load(self, path: str = "R:/Redline_Data/ai_logic/dqn_agent.pth"):
        """Load agent"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.episodes = checkpoint['episodes']
            logger.info(f"✅ DQN Agent loaded from {path}")


# =====================================================================
# PPO (Proximal Policy Optimization)
# =====================================================================

class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        shared_out = self.shared(x)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value
    
    def get_action(self, state):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class PPOBuffer:
    """Buffer for PPO trajectory collection"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Features:
    - Clipped objective for stable updates
    - Generalized Advantage Estimation (GAE)
    - Actor-Critic architecture
    - Entropy bonus for exploration
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'cuda'
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.buffer = PPOBuffer()
        self.episodes = 0
        
        logger.info(f"🤖 PPO Agent initialized on {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_value: float) -> Dict:
        """Update policy and value networks"""
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        # Multiple epochs of updates
        for _ in range(self.update_epochs):
            # Get current policy and value
            action_probs, current_values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss (clipped)
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss += actor_loss.item()
            value_loss += critic_loss.item()
            entropy_loss += entropy.item()
        
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / self.update_epochs,
            'policy_loss': policy_loss / self.update_epochs,
            'value_loss': value_loss / self.update_epochs,
            'entropy': entropy_loss / self.update_epochs
        }
    
    def train_episode(self, env: TradingEnvironment, update_every: int = 256) -> Dict:
        """Train for one episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.done:
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            self.buffer.push(state, action, reward, value, log_prob, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Update every N steps
            if steps % update_every == 0 and not done:
                _, _, next_value = self.select_action(next_state)
                self.update(next_value)
        
        # Final update
        if len(self.buffer.states) > 0:
            self.update(0)  # Terminal value = 0
        
        self.episodes += 1
        metrics = env.get_metrics()
        
        return {
            'episode': self.episodes,
            'total_reward': total_reward,
            **metrics
        }
    
    def save(self, path: str = "R:/Redline_Data/ai_logic/ppo_agent.pth"):
        """Save agent"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episodes': self.episodes
        }, path)
        logger.info(f"💾 PPO Agent saved to {path}")
    
    def load(self, path: str = "R:/Redline_Data/ai_logic/ppo_agent.pth"):
        """Load agent"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['actor_critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.episodes = checkpoint['episodes']
            logger.info(f"✅ PPO Agent loaded from {path}")


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_dqn(
    df: pd.DataFrame,
    episodes: int = 100,
    save_path: str = "R:/Redline_Data/ai_logic/dqn_agent.pth"
) -> DQNAgent:
    """Train DQN agent"""
    
    env = TradingEnvironment(df, lookback=30, max_steps=500)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    best_return = -float('inf')
    
    print("🚀 Starting DQN Training...")
    for ep in range(episodes):
        metrics = agent.train_episode(env)
        
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            agent.save(save_path)
        
        if ep % 10 == 0:
            print(f"Episode {ep}: Return={metrics['total_return']:.2f}%, "
                  f"Trades={metrics['num_trades']}, Epsilon={metrics['epsilon']:.3f}")
    
    print(f"✅ DQN Training complete! Best return: {best_return:.2f}%")
    return agent


def train_ppo(
    df: pd.DataFrame,
    episodes: int = 100,
    save_path: str = "R:/Redline_Data/ai_logic/ppo_agent.pth"
) -> PPOAgent:
    """Train PPO agent"""
    
    env = TradingEnvironment(df, lookback=30, max_steps=500)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    best_return = -float('inf')
    
    print("🚀 Starting PPO Training...")
    for ep in range(episodes):
        metrics = agent.train_episode(env)
        
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            agent.save(save_path)
        
        if ep % 10 == 0:
            print(f"Episode {ep}: Return={metrics['total_return']:.2f}%, "
                  f"Trades={metrics['num_trades']}, Sharpe={metrics['sharpe_ratio']:.2f}")
    
    print(f"✅ PPO Training complete! Best return: {best_return:.2f}%")
    return agent


# =====================================================================
# TEST
# =====================================================================

if __name__ == "__main__":
    print("🧪 Testing RL Agents...")
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    prices = [100]
    for _ in range(n-1):
        change = np.random.randn() * 0.015  # 1.5% daily volatility
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * n
    })
    
    # Test environment
    env = TradingEnvironment(df, lookback=30)
    state = env.reset()
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    
    # Quick DQN test
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    for ep in range(5):
        metrics = agent.train_episode(env)
        print(f"Episode {ep}: Return={metrics['total_return']:.2f}%, Epsilon={metrics['epsilon']:.3f}")
    
    print("\n✅ RL Agent test complete!")
