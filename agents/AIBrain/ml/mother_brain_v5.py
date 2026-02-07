"""
AIBrain v5.0 - Mother Brain with Temporal Fusion Transformer
=============================================================
State-of-the-art architecture with:
- Variable Selection Network (VSN) for feature importance
- LSTM Encoder for temporal patterns
- Interpretable Multi-Head Attention
- Gating mechanisms for noise reduction
- Multi-horizon prediction capability

Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Centralna konfiguracja
try:
    from config import (
        SEQ_LEN, TFT_HIDDEN_SIZE, TFT_NUM_HEADS, 
        TFT_LSTM_LAYERS, TFT_DROPOUT, TFT_TEMPORAL_FEATURES,
        TFT_CONTEXT_SIZE
    )
except ImportError:
    from agents.AIBrain.config import (
        SEQ_LEN, TFT_HIDDEN_SIZE, TFT_NUM_HEADS, 
        TFT_LSTM_LAYERS, TFT_DROPOUT, TFT_TEMPORAL_FEATURES,
        TFT_CONTEXT_SIZE
    )

from .child_agents import (
    ScannerAgent,
    TechnicianAgent,
    WhaleWatcherAgent,
    SentimentAgent,
    RugpullDetectorAgent,
    PortfolioManagerAgent,
    TrendAgent,
    MTFAgent,
    CorrelationAgent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MotherBrain_v5_TFT")


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for controlling information flow"""
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)
    Key component of TFT for processing features with skip connections
    """
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1, context_size=None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size or input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Context projection (if provided)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        # ELU activation
        self.elu = nn.ELU()
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # GLU for gating
        self.glu = GatedLinearUnit(hidden_size, self.output_size, dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)
        
        # Skip connection projection if sizes differ
        if input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context=None):
        # Layer 1
        hidden = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_fc(context)
        
        hidden = self.elu(hidden)
        
        # Layer 2
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # GLU gating
        gated = self.glu(hidden)
        
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x
        
        # Residual + normalization
        return self.layer_norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN)
    Learns which input features are most important
    """
    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.input_size = input_size
        
        # Single variable GRNs
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_inputs)
        ])
        
        # Variable selection GRN
        self.variable_selection_grn = GatedResidualNetwork(
            input_size * num_inputs, 
            hidden_size, 
            num_inputs, 
            dropout,
            context_size
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, context=None):
        """
        x: (batch, num_inputs, input_size) or (batch, seq, num_inputs, input_size)
        """
        # Handle different input dimensions
        if x.dim() == 3:
            batch_size, num_inputs, input_size = x.shape
            has_seq = False
        else:
            batch_size, seq_len, num_inputs, input_size = x.shape
            has_seq = True
            x = x.view(-1, num_inputs, input_size)
        
        # Process each variable through its GRN
        processed = []
        for i in range(self.num_inputs):
            var_output = self.single_variable_grns[i](x[:, i, :])
            processed.append(var_output)
        
        # Stack: (batch, num_inputs, hidden)
        processed = torch.stack(processed, dim=1)
        
        # Flatten for selection GRN: (batch, num_inputs * input_size)
        flat_x = x.view(x.size(0), -1)
        
        # Get variable selection weights
        selection_weights = self.variable_selection_grn(flat_x, context)
        selection_weights = self.softmax(selection_weights)
        
        # Weighted combination: (batch, hidden)
        combined = (processed * selection_weights.unsqueeze(-1)).sum(dim=1)
        
        if has_seq:
            combined = combined.view(batch_size, seq_len, -1)
            selection_weights = selection_weights.view(batch_size, seq_len, -1)
        
        return combined, selection_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention
    Modified attention that provides interpretable attention weights
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        # Average attention across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)
        
        return output, avg_attn_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for Trading
    
    Architecture:
    1. Variable Selection Networks for feature processing
    2. LSTM Encoder/Decoder for sequence modeling
    3. Interpretable Multi-Head Attention for temporal patterns
    4. Gating layers for noise reduction
    5. Output heads for classification
    """
    
    def __init__(
        self,
        num_agents: int = 8,
        temporal_features: int = 6,
        context_size: int = 10,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        sequence_length: int = 30,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.temporal_features = temporal_features
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # === 1. INPUT EMBEDDINGS ===
        # Project temporal features to hidden size
        self.temporal_embedding = nn.Linear(temporal_features, hidden_size)
        
        # Project agent signals to hidden size
        self.agent_embedding = nn.Linear(num_agents, hidden_size)
        
        # Project context to hidden size
        self.context_embedding = nn.Linear(context_size, hidden_size)
        
        # === 2. VARIABLE SELECTION NETWORKS ===
        # For static context (agent signals + market context)
        self.static_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=num_agents + context_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # For temporal inputs (price sequence)
        self.temporal_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=temporal_features,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # === 3. LSTM ENCODER ===
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # === 4. STATIC ENRICHMENT ===
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout, context_size=hidden_size
        )
        
        # === 5. INTERPRETABLE MULTI-HEAD ATTENTION ===
        self.self_attention = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout
        )
        
        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        
        # === 6. POSITION-WISE FEED-FORWARD ===
        self.ff_grn = GatedResidualNetwork(
            hidden_size, hidden_size * 2, hidden_size, dropout
        )
        
        # === 7. OUTPUT LAYERS ===
        self.pre_output = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.num_quantiles)  # Quantiles (e.g. 0.1, 0.5, 0.9)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Attention output for interpretability
        self.attention_output = nn.Linear(hidden_size, num_agents)
        
    def forward(
        self,
        agent_signals: torch.Tensor,  # (batch, num_agents)
        market_context: torch.Tensor,  # (batch, context_size)
        price_sequence: torch.Tensor = None  # (batch, seq_len, temporal_features)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with multi-quantile outputs
        
        Returns:
            quantiles: (batch, num_quantiles) - e.g. [q0.1, q0.5, q0.9] returns
            attention_weights: (batch, num_agents) - which agents were important
            interpretability: dict with attention maps
        """
        batch_size = agent_signals.size(0)
        interpretability = {}
        
        # === STATIC PROCESSING ===
        # Combine agent signals and context
        static_inputs = torch.cat([agent_signals, market_context], dim=-1)
        static_inputs = static_inputs.unsqueeze(-1)  # (batch, num_static, 1)
        
        # Variable selection for static inputs
        static_features, static_weights = self.static_vsn(static_inputs)
        interpretability['static_selection'] = static_weights
        
        # === TEMPORAL PROCESSING ===
        if price_sequence is not None:
            seq_len = price_sequence.size(1)
            
            # Reshape for VSN: (batch, seq, features, 1)
            temporal_inputs = price_sequence.unsqueeze(-1)
            
            # Process each timestep
            temporal_features_list = []
            temporal_weights_list = []
            
            for t in range(seq_len):
                t_input = temporal_inputs[:, t, :, :]  # (batch, features, 1)
                t_feat, t_weights = self.temporal_vsn(t_input)
                temporal_features_list.append(t_feat)
                temporal_weights_list.append(t_weights)
            
            temporal_features = torch.stack(temporal_features_list, dim=1)  # (batch, seq, hidden)
            temporal_weights = torch.stack(temporal_weights_list, dim=1)
            interpretability['temporal_selection'] = temporal_weights
            
            # LSTM encoding
            lstm_out, (h_n, c_n) = self.lstm_encoder(temporal_features)
            
            # Use last hidden state
            encoded = lstm_out[:, -1, :]  # (batch, hidden)
        else:
            # No temporal data - use static features only
            encoded = self.context_embedding(market_context)
        
        # === STATIC ENRICHMENT ===
        # Enrich encoded features with static context
        enriched = self.static_enrichment(encoded, static_features)
        
        # === SELF-ATTENTION ===
        # Reshape for attention if we have sequence
        if price_sequence is not None:
            # Use LSTM output sequence for attention
            attended, attn_weights = self.self_attention(lstm_out, lstm_out, lstm_out)
            interpretability['temporal_attention'] = attn_weights
            
            # Post-attention processing
            attended = self.post_attention_grn(attended[:, -1, :])
        else:
            attended = enriched
        
        # === FEED-FORWARD ===
        output = self.ff_grn(attended + enriched)
        output = self.layer_norm(output)
        
        # === OUTPUT ===
        pre_out = F.relu(self.pre_output(output))
        quantiles = self.output_layer(pre_out)  # Multi-quantile predictions
        
        # Agent attention weights (interpretable)
        agent_attention = torch.softmax(self.attention_output(output), dim=-1)
        
        return quantiles, agent_attention, interpretability


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss)
    L(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: (batch, num_quantiles)
        target: (batch) or (batch, 1)
        """
        if target.dim() == 1:
            target = target.unsqueeze(1)
            
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        combined_loss = torch.stack(losses, dim=1).sum(dim=1)
        return combined_loss.mean()


class MotherBrainV5:
    """
    Mother Brain v5.0 - TFT Edition
    
    Features:
    - Temporal Fusion Transformer architecture
    - Interpretable attention (know WHY the model decided)
    - Variable selection (know WHICH features matter)
    - 8 child agents including new TrendAgent and MTFAgent
    - ~100K+ parameters
    """
    
    def __init__(
        self,
        num_agents: int = 9,
        context_size: int = TFT_CONTEXT_SIZE,
        hidden_size: int = 64,
        device: str = 'cuda',
        learning_rate: float = 0.0005
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"🧠 Mother Brain v5.0 (TFT) initialized on {self.device}")
        
        self.num_agents = num_agents
        self.context_size = context_size
        
        # Initialize TFT model
        self.model = TemporalFusionTransformer(
            num_agents=num_agents,
            temporal_features=TFT_TEMPORAL_FEATURES,
            context_size=context_size,
            hidden_size=hidden_size,
            lstm_layers=TFT_LSTM_LAYERS,
            num_heads=TFT_NUM_HEADS,
            dropout=TFT_DROPOUT,
            sequence_length=SEQ_LEN
        ).to(self.device)
        
        # Loss & Optimizer
        self.criterion = QuantileLoss(self.model.quantiles)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Log architecture
        self._log_architecture()
        
        # Initialize child agents (8 agents now)
        self.children = {}
        self._spawn_children()
        
        # Training stats
        self.total_steps = 0
        self.best_accuracy = 0.0

    def parameters(self):
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
    
    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("📊 TFT Architecture Summary:")
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,}")
        logger.info(f"   Hidden Size: 64")
        logger.info(f"   Attention Heads: 4")
        logger.info(f"   LSTM Layers: 2")
        logger.info(f"   Agents: {self.num_agents}")
    
    def _spawn_children(self):
        """Spawn all 9 child agents"""
        self.children = {
            'scanner': ScannerAgent(),
            'technician': TechnicianAgent(),
            'whale': WhaleWatcherAgent(),
            'sentiment': SentimentAgent(),
            'rugpull': RugpullDetectorAgent(),
            'portfolio': PortfolioManagerAgent(),
            'trend': TrendAgent(),
            'mtf': MTFAgent(),
            'correlation': CorrelationAgent()
        }
        logger.info(f"👶 Spawned {len(self.children)} child agents")
    
    async def collect_signals(self, market_data: dict) -> List[float]:
        """Collect signals from all child agents"""
        signals = []
        for name, agent in self.children.items():
            try:
                await agent.analyze(market_data)
                signal = agent.get_signal_for_attention()
                signals.append(signal)
            except Exception as e:
                signals.append(0.0)
        return signals
    
    def prepare_sequence(self, df, seq_len: int = 30) -> torch.Tensor:
        """Prepare price sequence for TFT"""
        if df is None or len(df) < seq_len:
            return None
        
        try:
            # Get last seq_len candles
            recent = df.tail(seq_len)
            
            # Normalize features
            close = recent['close'].values
            close_norm = (close - close[0]) / close[0] if close[0] != 0 else close
            
            # Create feature matrix: [close_norm, returns, high_low_range, volume_norm, rsi, volatility]
            features = np.zeros((seq_len, 6))
            features[:, 0] = close_norm
            features[:, 1] = np.diff(close, prepend=close[0]) / close[0]  # Returns
            features[:, 2] = (recent['high'].values - recent['low'].values) / close  # Range
            features[:, 3] = recent['volume'].values / recent['volume'].mean()  # Volume norm
            
            # RSI proxy (simplified)
            gains = np.maximum(np.diff(close, prepend=close[0]), 0)
            losses = np.maximum(-np.diff(close, prepend=close[0]), 0)
            avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
            avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
            rs = avg_gain / (avg_loss + 1e-8)
            features[:, 4] = (100 - (100 / (1 + rs))) / 100  # Normalized RSI
            
            # Volatility (rolling std)
            features[:, 5] = np.convolve(np.abs(features[:, 1]), np.ones(5)/5, mode='same')
            
            # Handle NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
        except Exception as e:
            logger.error(f"Sequence preparation error: {e}")
            return None
    
    def calculate_confidence(self, quantiles: torch.Tensor) -> float:
        """
        Calculates confidence score based on the spread of quantiles.
        Lower spread (q0.9 - q0.1) relative to median (q0.5) means higher confidence.
        
        Returns:
            confidence: 0.0 to 1.0
        """
        if quantiles.dim() == 1:
            quantiles = quantiles.unsqueeze(0)
            
        # Assuming quantiles is [q0.1, q0.5, q0.9]
        q01, q05, q09 = quantiles[0, 0], quantiles[0, 1], quantiles[0, 2]
        
        spread = torch.abs(q09 - q01)
        # Normalize spread - higher spread = lower confidence
        # We use a simple sigmoid-like normalization or exponential decay
        confidence = torch.exp(-spread * 50).item() # Tuning factor 50 for percentage-based returns
        
        return min(max(confidence, 0.0), 1.0)

    def train_step(
        self,
        agent_signals: List[float],
        market_context: List[float],
        correct_return: float, # Changed from 'correct_action'
        price_sequence: torch.Tensor = None
    ) -> Tuple[float, Dict]:
        """
        One training step
        
        Returns:
            loss: float
            interpretability: Dict with feature importance, attention weights, etc.
        """
        self.model.train()
        
        # Pad signals if needed
        while len(agent_signals) < self.num_agents:
            agent_signals.append(0.0)
        agent_signals = agent_signals[:self.num_agents]
        
        while len(market_context) < self.context_size:
            market_context.append(0.0)
        market_context = market_context[:self.context_size]
        
        # Convert to tensors
        signals_tensor = torch.FloatTensor([agent_signals]).to(self.device)
        context_tensor = torch.FloatTensor([market_context]).to(self.device)
        target = torch.FloatTensor([correct_return]).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        quantiles, agent_attention, interpretability = self.model(
            signals_tensor, context_tensor, price_sequence
        )
        
        # Loss and backward
        loss = self.criterion(quantiles, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.total_steps += 1
        
        # Return interpretability info
        return loss.item(), {
            'agent_attention': agent_attention.detach().cpu().numpy()[0],
            'prediction': quantiles.detach().cpu().numpy()[0], # Returns [q0.1, q0.5, q0.9]
            'confidence': self.calculate_confidence(quantiles.detach().cpu())
        }
    
    def predict(
        self,
        agent_signals: List[float],
        market_context: List[float],
        price_sequence: torch.Tensor = None
    ) -> Tuple[int, np.ndarray, Dict]:
        """
        Make prediction with interpretable outputs
        
        Returns:
            action: 0=BUY, 1=HOLD, 2=SELL
            agent_attention: importance of each agent
            interpretability: detailed interpretation
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare inputs
            while len(agent_signals) < self.num_agents:
                agent_signals.append(0.0)
            signals_tensor = torch.FloatTensor([agent_signals[:self.num_agents]]).to(self.device)
            
            while len(market_context) < self.context_size:
                market_context.append(0.0)
            context_tensor = torch.FloatTensor([market_context[:self.context_size]]).to(self.device)
            # Forward pass
            quantiles, agent_attention, interpretability = self.model(
                signals_tensor, context_tensor, price_sequence
            )
            
            q_values = quantiles.detach().cpu().numpy()[0]
            # Action logic: 0=HOLD, 1=BUY, 2=SELL
            # q_values[1] is the median return (q0.5)
            median_return = q_values[1]
            
            threshold = 0.005 # 0.5% return threshold
            if median_return > threshold:
                action = 1 # BUY
            elif median_return < -threshold:
                action = 2 # SELL
            else:
                action = 0 # HOLD
                
            confidence = self.calculate_confidence(quantiles.detach().cpu())
            
            return action, agent_attention.detach().cpu().numpy()[0], {
                'quantiles': q_values,
                'confidence': confidence,
                'median_return': median_return,
                **interpretability
            }
    
    def save(self, path: str = "R:/Redline_Data/ai_logic/mother_v5_tft.pth"):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'best_accuracy': self.best_accuracy,
            'acc': self.best_accuracy  # Dual key for compatibility
        }, path)
        logger.info(f"💾 Mother Brain v5 (TFT) saved to {path}")
    
    def load(self, path: str = "R:/Redline_Data/ai_logic/mother_v5_tft.pth"):
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_accuracy = checkpoint.get('best_accuracy', checkpoint.get('acc', 0.0))
            logger.info(f"✅ Mother Brain v5 (TFT) loaded from {path}")
            return True
        return False
    
    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts for reporting"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable
        }
