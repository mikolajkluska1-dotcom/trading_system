"""
AIBrain v3.0 - Mother Brain with Dynamic Attention Mechanism
The Merger - combines existing v2 weights with expanded architecture
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
logger = logging.getLogger("MotherBrain_v3")


class AttentionRouterV3(nn.Module):
    """
    v3.0 Attention Router with dynamic agent count
    Can load weights from v2 (6 agents) and expand to more
    """
    
    def __init__(self, num_agents=6, context_size=10, hidden_size=64):
        super(AttentionRouterV3, self).__init__()
        
        self.num_agents = num_agents
        self.context_size = context_size
        
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
        weights = self.attention_net(market_context)
        weighted_signals = agent_signals * weights
        combined = torch.cat((weighted_signals, market_context), dim=1)
        logits = self.decision_net(combined)
        return logits, weights


class MotherBrainV3:
    """
    Mother Brain v3.0 - The Merger
    
    Features:
    - Dynamic agent count (expandable)
    - Migrates weights from v2
    - Real child agent integration
    - VETO system for risk management
    - GPU acceleration
    """
    
    def __init__(self, num_agents=6, context_size=10, device='cuda', learning_rate=0.003):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        print(f"🧠 Mother Brain v3.0 initialized on {self.device}")
        
        self.num_agents = num_agents
        self.context_size = context_size
        
        self.router = AttentionRouterV3(num_agents, context_size).to(self.device)
        self.optimizer = optim.Adam(self.router.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Data recorder
        self.recorder = BrainRecorder()
        
        # Initialize child agents
        self.children = {}
        self._spawn_children()
        
        # Performance tracking
        self.total_decisions = 0
        self.correct_decisions = 0
    
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
    
    def register_child(self, name, agent_instance):
        """Register additional child agent"""
        self.children[name] = agent_instance
        logger.info(f"📎 Registered child agent: {name}")
    
    async def collect_signals(self, market_data: dict) -> list:
        """
        Collect signals from all child agents asynchronously
        
        Returns:
            list of floats (scores from each agent)
        """
        agent_order = ['scanner', 'technician', 'whale_watcher', 'sentiment', 'rugpull_detector', 'portfolio_manager']
        
        # Run all analyses
        tasks = []
        for name in agent_order:
            agent = self.children.get(name)
            if agent:
                tasks.append(agent.analyze(market_data))
        
        await asyncio.gather(*tasks)
        
        # Collect scores
        scores = []
        for name in agent_order:
            agent = self.children.get(name)
            if agent:
                scores.append(agent.get_signal_for_attention())
            else:
                scores.append(0.0)
        
        return scores
    
    def make_decision(self, agent_signals: list, market_context: dict) -> dict:
        """
        Make a trading decision using agent signals and market context
        
        Args:
            agent_signals: list of floats from child agents
            market_context: dict with volatility, btc_trend, etc.
        
        Returns:
            dict with 'action', 'confidence', 'attention_weights'
        """
        # Build context vector
        ctx_list = [
            market_context.get('volatility', 0),
            market_context.get('btc_trend', 0),
            market_context.get('funding_rate', 0),
            market_context.get('fear_greed', 50) / 100.0,
            market_context.get('long_short_ratio', 1.0) - 1,
            market_context.get('volume_change', 0),
            0, 0, 0, 0  # Padding to context_size
        ][:self.context_size]
        
        # Ensure correct number of agents
        while len(agent_signals) < self.num_agents:
            agent_signals.append(0.0)
        agent_signals = agent_signals[:self.num_agents]
        
        # Convert to tensors
        signals_tensor = torch.tensor([agent_signals], dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor([ctx_list], dtype=torch.float32).to(self.device)
        
        # Inference
        self.router.eval()
        with torch.no_grad():
            logits, weights = self.router(signals_tensor, context_tensor)
            probs = torch.softmax(logits, dim=1)
        
        action_idx = torch.argmax(probs).item()
        actions = ['BUY', 'HOLD', 'SELL']
        final_action = actions[action_idx]
        confidence = probs[0][action_idx].item()
        
        # VETO Checks
        veto_reason = None
        
        # Rugpull detector VETO
        rugpull_agent = self.children.get('rugpull_detector')
        if rugpull_agent and rugpull_agent.last_analysis.get('signal') == 'VETO':
            final_action = 'HOLD'
            veto_reason = rugpull_agent.last_analysis.get('reasoning', 'Rugpull risk')
            logger.warning(f"⛔ VETO: {veto_reason}")
        
        # Portfolio manager VETO for buys
        pm_agent = self.children.get('portfolio_manager')
        if pm_agent and pm_agent.last_analysis.get('signal') == 'VETO' and final_action == 'BUY':
            final_action = 'HOLD'
            veto_reason = pm_agent.last_analysis.get('reasoning', 'Portfolio limits')
            logger.warning(f"⛔ VETO: {veto_reason}")
        
        self.total_decisions += 1
        
        return {
            'action': final_action,
            'confidence': confidence,
            'attention_weights': weights.cpu().numpy().tolist()[0],
            'veto_reason': veto_reason,
            'probabilities': probs.cpu().numpy().tolist()[0]
        }
    
    def train_step(self, agent_signals: list, market_context: list, correct_action: int) -> float:
        """
        One supervised training step
        
        Args:
            agent_signals: list of agent scores
            market_context: list of context features
            correct_action: 0=BUY, 1=HOLD, 2=SELL
        
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
        logits, _ = self.router(signals_tensor, context_tensor)
        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def migrate_from_v2(self, v2_path: str = "R:/Redline_Data/ai_logic/mother_v2.pth") -> bool:
        """
        Migrate weights from v2 model (6 agents) to v3
        
        Args:
            v2_path: path to v2 checkpoint
        
        Returns:
            True if successful
        """
        if not os.path.exists(v2_path):
            logger.warning(f"V2 model not found at {v2_path}")
            return False
        
        try:
            checkpoint = torch.load(v2_path, map_location=self.device)
            old_state = checkpoint['model_state']
            
            # Get current state
            new_state = self.router.state_dict()
            
            # Copy compatible weights
            for key in old_state:
                if key in new_state:
                    old_shape = old_state[key].shape
                    new_shape = new_state[key].shape
                    
                    if old_shape == new_shape:
                        # Same shape - direct copy
                        new_state[key] = old_state[key]
                    elif len(old_shape) == 2 and len(new_shape) == 2:
                        # Different shape - partial copy (for expanded agents)
                        min_rows = min(old_shape[0], new_shape[0])
                        min_cols = min(old_shape[1], new_shape[1])
                        new_state[key][:min_rows, :min_cols] = old_state[key][:min_rows, :min_cols]
                        logger.info(f"Partial copy for {key}: {old_shape} -> {new_shape}")
                    elif len(old_shape) == 1 and len(new_shape) == 1:
                        # Bias vectors
                        min_size = min(old_shape[0], new_shape[0])
                        new_state[key][:min_size] = old_state[key][:min_size]
            
            self.router.load_state_dict(new_state)
            logger.info(f"✅ Migrated weights from v2 to v3")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def save(self, path: str = "R:/Redline_Data/ai_logic/mother_v3.pth"):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.router.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'num_agents': self.num_agents,
            'context_size': self.context_size,
            'total_decisions': self.total_decisions,
            'version': '3.0'
        }, path)
        logger.info(f"💾 Mother Brain v3 saved to {path}")
    
    def load(self, path: str = "R:/Redline_Data/ai_logic/mother_v3.pth") -> bool:
        """Load model checkpoint"""
        if not os.path.exists(path):
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check architecture compatibility
            saved_agents = checkpoint.get('num_agents', 6)
            saved_context = checkpoint.get('context_size', 10)
            
            if saved_agents != self.num_agents or saved_context != self.context_size:
                logger.warning(f"Architecture mismatch: saved({saved_agents}, {saved_context}) vs current({self.num_agents}, {self.context_size})")
                # Attempt partial load
                return self.migrate_from_v2(path)
            
            self.router.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.total_decisions = checkpoint.get('total_decisions', 0)
            
            logger.info(f"📂 Mother Brain v3 loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False
    
    def get_child_status(self) -> dict:
        """Get status of all child agents"""
        status = {}
        for name, agent in self.children.items():
            status[name] = {
                'last_signal': agent.last_analysis.get('signal', 'UNKNOWN'),
                'last_score': agent.last_analysis.get('score', 0),
                'confidence': agent.last_analysis.get('confidence', 0)
            }
        return status
