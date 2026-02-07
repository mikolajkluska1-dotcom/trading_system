"""
AIBrain v2.0 - Mother Brain with Attention Mechanism
The central AI decision maker with dynamic agent attention weights
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from .recorder import BrainRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MotherBrain")


class AttentionRouter(nn.Module):
    """Neural network that learns which agents to trust in different market conditions"""
    
    def __init__(self, num_agents=6, context_size=10, hidden_size=64):
        super(AttentionRouter, self).__init__()
        
        # Attention Gate - learns which agents to focus on
        self.attention_net = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_agents),
            nn.Softmax(dim=1)
        )
        
        # Decision Head - makes final trading decision
        self.decision_net = nn.Sequential(
            nn.Linear(num_agents + context_size, hidden_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # BUY, HOLD, SELL
        )

    def forward(self, agent_signals, market_context):
        # Calculate attention weights based on market context
        weights = self.attention_net(market_context)
        
        # Weight the agent signals
        weighted_signals = agent_signals * weights
        
        # Combine weighted signals with context for final decision
        combined = torch.cat((weighted_signals, market_context), dim=1)
        logits = self.decision_net(combined)
        
        return logits, weights


class MotherBrain:
    """
    Mother Brain v2.0 - Central Decision Maker
    
    Features:
    - Attention mechanism to dynamically weight child agents
    - VETO system for risk management
    - Decision recording for training data
    - GPU acceleration support
    """
    
    def __init__(self, num_agents=6, context_size=10, device='cuda'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"🧠 Mother Brain v2.0 initialized on {self.device}")
        
        self.router = AttentionRouter(num_agents, context_size).to(self.device)
        self.optimizer = optim.Adam(self.router.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data recorder for training
        self.recorder = BrainRecorder()
        
        # Child agents registry
        self.children = {}
        
        # Performance tracking
        self.total_decisions = 0
        self.correct_decisions = 0

    def register_child(self, name, agent_instance):
        """Register a child agent"""
        self.children[name] = agent_instance
        logger.info(f"📎 Registered child agent: {name}")

    def make_decision(self, market_data, market_context_dict):
        """
        Make a trading decision using all child agents and attention mechanism
        
        Args:
            market_data: dict with symbol, price, etc.
            market_context_dict: dict with volatility, btc_trend, funding_rate, etc.
        
        Returns:
            dict with 'action' and 'confidence'
        """
        # 1. Collect signals from all agents (fixed order for consistency)
        agent_order = ['scanner', 'technician', 'whale_watcher', 'sentiment', 'rugpull_detector', 'portfolio_manager']
        agent_scores = []
        agent_signals_dict = {}
        
        for name in agent_order:
            agent = self.children.get(name)
            score = 0.0
            
            if agent and hasattr(agent, 'last_analysis'):
                raw = agent.last_analysis.get('score', 0)
                sig = agent.last_analysis.get('signal', 'NEUTRAL')
                
                # VETO signals get strong negative score
                if sig == 'VETO':
                    raw = -1.0
                    
                score = float(raw)
            
            agent_scores.append(score)
            agent_signals_dict[name] = score

        # 2. Convert to tensors
        signals_tensor = torch.tensor([agent_scores], dtype=torch.float32).to(self.device)
        
        # Build context vector (pad to context_size)
        ctx_list = [
            market_context_dict.get('volatility', 0),
            market_context_dict.get('btc_trend', 0),
            market_context_dict.get('funding_rate', 0),
            market_context_dict.get('fear_greed', 50) / 100.0,
            market_context_dict.get('volume_change', 0),
            0, 0, 0, 0, 0  # Padding to 10 features
        ]
        context_tensor = torch.tensor([ctx_list[:10]], dtype=torch.float32).to(self.device)

        # 3. AI Inference
        self.router.eval()
        with torch.no_grad():
            logits, weights = self.router(signals_tensor, context_tensor)
            probs = torch.softmax(logits, dim=1)
            
        action_idx = torch.argmax(probs).item()
        actions = ['BUY', 'HOLD', 'SELL']
        final_action = actions[action_idx]
        confidence = probs[0][action_idx].item()

        # 4. VETO Checks (hard overrides)
        veto_reason = None
        
        # Rugpull detector VETO
        if agent_signals_dict.get('rugpull_detector', 0) <= -0.5:
            final_action = 'HOLD'
            veto_reason = "Rugpull risk detected"
            print(f"⛔ VETO: {veto_reason}")

        # Portfolio manager VETO for buys
        if agent_signals_dict.get('portfolio_manager', 0) <= -0.5 and final_action == 'BUY':
            final_action = 'HOLD'
            veto_reason = "Portfolio limits exceeded"
            print(f"⛔ VETO: {veto_reason}")

        # 5. Record decision for training
        self.recorder.record_decision(
            symbol=market_data.get('symbol', 'UNKNOWN'),
            agent_signals=agent_signals_dict,
            attention_weights=weights.cpu().numpy().tolist()[0],
            market_context=market_context_dict,
            final_decision={'action': final_action, 'confidence': confidence}
        )
        
        self.total_decisions += 1

        return {
            'action': final_action, 
            'confidence': confidence,
            'attention_weights': weights.cpu().numpy().tolist()[0],
            'veto_reason': veto_reason
        }

    def train_step(self, agent_signals, market_context, correct_action):
        """
        One training step with labeled data
        
        Args:
            agent_signals: list of 6 agent scores
            market_context: list of 10 context features
            correct_action: 0=BUY, 1=HOLD, 2=SELL
        """
        self.router.train()
        
        signals_tensor = torch.tensor([agent_signals], dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor([market_context], dtype=torch.float32).to(self.device)
        target = torch.tensor([correct_action], dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        logits, _ = self.router(signals_tensor, context_tensor)
        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, path="R:/Redline_Data/ai_logic/mother_v2.pth"):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.router.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_decisions': self.total_decisions
        }, path)
        print(f"💾 Mother Brain saved to {path}")
        
    def load(self, path="R:/Redline_Data/ai_logic/mother_v2.pth"):
        """Load model weights"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.router.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.total_decisions = checkpoint.get('total_decisions', 0)
            print(f"📂 Mother Brain loaded from {path}")
            return True
        return False
    
    def get_attention_stats(self):
        """Get statistics about which agents are being trusted most"""
        return {
            'total_decisions': self.total_decisions,
            'registered_agents': list(self.children.keys())
        }
