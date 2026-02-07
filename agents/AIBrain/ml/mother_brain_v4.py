"""
AIBrain v4.0 - Mother Brain with LSTM Temporal Memory
=====================================================
Advanced architecture with:
- LSTM for temporal pattern recognition (30 candles)
- Deep attention mechanism  
- ~50K parameters
- GPU optimized
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import asyncio
from .recorder import BrainRecorder
from .child_agents import (
    ScannerAgent,
    TechnicianAgent,
    WhaleWatcherAgent,
    SentimentAgent,
    RugpullDetectorAgent,
    PortfolioManagerAgent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MotherBrain_v4")


class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder for price sequence analysis.
    Looks at last 30 candles to find patterns.
    """
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2):
        super(TemporalEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,      # OHLCV + RSI
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        x: (batch, seq_len, features) - e.g., (1, 30, 6)
        returns: (batch, 32) - temporal features
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Project to temporal features
        out = self.fc(last_hidden)
        out = self.relu(out)
        
        return out


class AttentionRouterV4(nn.Module):
    """
    v4.0 Attention Router with LSTM Temporal Memory
    
    Architecture:
    - Temporal Encoder (LSTM): 30 candles → 32 features
    - Deep Attention Gate: context → agent weights
    - Deep Decision Head: combined → BUY/HOLD/SELL
    
    Total params: ~50K
    """
    
    def __init__(self, num_agents=6, context_size=10, hidden_size=128, 
                 sequence_length=30, temporal_features=6):
        super(AttentionRouterV4, self).__init__()
        
        self.num_agents = num_agents
        self.context_size = context_size
        self.sequence_length = sequence_length
        
        # 1. Temporal Encoder (LSTM)
        self.temporal_encoder = TemporalEncoder(
            input_size=temporal_features,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        # 2. Deep Attention Gate (using LayerNorm for batch_size=1 compatibility)
        # Input: context (10) + temporal features (32) = 42
        attention_input = context_size + 32
        self.attention_net = nn.Sequential(
            nn.Linear(attention_input, 64),
            nn.LayerNorm(64),  # LayerNorm works with batch_size=1
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_agents),
            nn.Softmax(dim=1)
        )
        
        # 3. Deep Decision Head (using LayerNorm)
        # Input: agents (6) + context (10) + temporal (32) = 48
        decision_input = num_agents + context_size + 32
        self.decision_net = nn.Sequential(
            nn.Linear(decision_input, hidden_size),
            nn.LayerNorm(hidden_size),  # LayerNorm works with batch_size=1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # BUY, HOLD, SELL
        )
    
    def forward(self, agent_signals, market_context, price_sequence=None):
        """
        Forward pass with optional temporal sequence.
        
        Args:
            agent_signals: (batch, num_agents)
            market_context: (batch, context_size)
            price_sequence: (batch, seq_len, features) - optional OHLCV sequence
        
        Returns:
            logits: (batch, 3)
            attention_weights: (batch, num_agents)
        """
        batch_size = agent_signals.size(0)
        
        # 1. Temporal encoding (if sequence provided)
        if price_sequence is not None:
            temporal_features = self.temporal_encoder(price_sequence)
        else:
            # Fallback: zero temporal features
            temporal_features = torch.zeros(batch_size, 32, device=agent_signals.device)
        
        # 2. Attention computation
        attention_input = torch.cat([market_context, temporal_features], dim=1)
        attention_weights = self.attention_net(attention_input)
        
        # 3. Weight agent signals
        weighted_signals = agent_signals * attention_weights
        
        # 4. Decision
        decision_input = torch.cat([weighted_signals, market_context, temporal_features], dim=1)
        logits = self.decision_net(decision_input)
        
        return logits, attention_weights


class MotherBrainV4:
    """
    Mother Brain v4.0 - LSTM Temporal Memory Edition
    
    Features:
    - LSTM for pattern recognition in price sequences
    - ~50K parameters (vs 1.8K in v3)
    - Deep attention mechanism
    - Dropout + BatchNorm for regularization
    - GPU acceleration
    """
    
    def __init__(self, num_agents=6, context_size=10, device='cuda', learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"🧠 Mother Brain v4.0 (LSTM) initialized on {self.device}")
        
        self.num_agents = num_agents
        self.context_size = context_size
        
        self.router = AttentionRouterV4(num_agents, context_size).to(self.device)
        self.optimizer = optim.Adam(self.router.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Data recorder
        self.recorder = BrainRecorder()
        
        # Initialize child agents
        self.children = {}
        self._spawn_children()
        
        # Performance tracking
        self.total_decisions = 0
        self.correct_decisions = 0
        
        # Log architecture
        self._log_architecture()
    
    def _log_architecture(self):
        """Log model architecture details"""
        total_params = sum(p.numel() for p in self.router.parameters())
        trainable_params = sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        
        logger.info(f"📊 Architecture Summary:")
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,}")
        logger.info(f"   LSTM Layers: 2")
        logger.info(f"   Hidden Size: 128")
        logger.info(f"   Sequence Length: 30 candles")
    
    def _spawn_children(self):
        """Spawn all child agents"""
        self.children = {
            'scanner': ScannerAgent(),
            'technician': TechnicianAgent(),
            'whale_watcher': WhaleWatcherAgent(),
            'sentiment': SentimentAgent(),
            'rugpull_detector': RugpullDetectorAgent(),
            'portfolio_manager': PortfolioManagerAgent()
        }
        logger.info(f"👶 Spawned {len(self.children)} child agents")
    
    async def collect_signals(self, market_data: dict) -> list:
        """Collect signals from all child agents asynchronously"""
        agent_order = ['scanner', 'technician', 'whale_watcher', 'sentiment', 'rugpull_detector', 'portfolio_manager']
        
        tasks = []
        for name in agent_order:
            agent = self.children.get(name)
            if agent:
                tasks.append(agent.analyze(market_data))
        
        await asyncio.gather(*tasks)
        
        scores = []
        for name in agent_order:
            agent = self.children.get(name)
            if agent:
                scores.append(agent.get_signal_for_attention())
            else:
                scores.append(0.0)
        
        return scores
    
    def prepare_sequence(self, df, seq_len=30) -> torch.Tensor:
        """
        Prepare price sequence for LSTM from DataFrame.
        
        Args:
            df: DataFrame with OHLCV + RSI
            seq_len: sequence length (default 30)
        
        Returns:
            tensor: (1, seq_len, 6)
        """
        if df is None or len(df) < seq_len:
            return None
        
        # Take last seq_len candles
        df_seq = df.tail(seq_len).copy()
        
        # Normalize features (simple min-max per sequence)
        features = []
        
        # OHLC (normalized by close)
        close = df_seq['close'].values
        base = close[0] if close[0] > 0 else 1
        
        features.append((df_seq['open'].values / base - 1) * 100)   # Open %
        features.append((df_seq['high'].values / base - 1) * 100)   # High %
        features.append((df_seq['low'].values / base - 1) * 100)    # Low %
        features.append((close / base - 1) * 100)                    # Close %
        
        # Volume (normalized)
        vol = df_seq['volume'].values
        vol_norm = vol / (vol.mean() + 1e-8)
        features.append(vol_norm)
        
        # RSI (already 0-100, normalize to -1 to 1)
        if 'rsi' in df_seq.columns:
            rsi = (df_seq['rsi'].values - 50) / 50
        else:
            rsi = np.zeros(seq_len)
        features.append(rsi)
        
        # Stack: (seq_len, 6)
        sequence = np.stack(features, axis=1).astype(np.float32)
        
        # To tensor: (1, seq_len, 6)
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return tensor
    
    def train_step(self, agent_signals: list, market_context: list, 
                   correct_action: int, price_sequence=None) -> float:
        """
        One supervised training step with optional sequence.
        
        Args:
            agent_signals: list of agent scores
            market_context: list of context features
            correct_action: 0=BUY, 1=HOLD, 2=SELL
            price_sequence: optional (1, seq_len, 6) tensor
        
        Returns:
            loss value
        """
        self.router.train()
        
        # Pad signals if needed
        while len(agent_signals) < self.num_agents:
            agent_signals.append(0.0)
        agent_signals = agent_signals[:self.num_agents]
        
        while len(market_context) < self.context_size:
            market_context.append(0.0)
        market_context = market_context[:self.context_size]
        
        signals_tensor = torch.tensor([agent_signals], dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor([market_context], dtype=torch.float32).to(self.device)
        target = torch.tensor([correct_action], dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        logits, _ = self.router(signals_tensor, context_tensor, price_sequence)
        loss = self.criterion(logits, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str = "R:/Redline_Data/ai_logic/mother_v4.pth"):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.router.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'num_agents': self.num_agents,
            'context_size': self.context_size,
            'total_decisions': self.total_decisions,
            'version': '4.0-LSTM'
        }, path)
        logger.info(f"💾 Mother Brain v4 (LSTM) saved to {path}")
    
    def load(self, path: str = "R:/Redline_Data/ai_logic/mother_v4.pth") -> bool:
        """Load model checkpoint"""
        if not os.path.exists(path):
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.router.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.total_decisions = checkpoint.get('total_decisions', 0)
            
            logger.info(f"📂 Mother Brain v4 (LSTM) loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
    
    def get_param_count(self) -> dict:
        """Get parameter counts for reporting"""
        total = sum(p.numel() for p in self.router.parameters())
        trainable = sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        
        # Per component
        temporal = sum(p.numel() for p in self.router.temporal_encoder.parameters())
        attention = sum(p.numel() for p in self.router.attention_net.parameters())
        decision = sum(p.numel() for p in self.router.decision_net.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'temporal_encoder': temporal,
            'attention_net': attention,
            'decision_net': decision
        }


# For backward compatibility with training scripts
import numpy as np
