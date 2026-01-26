"""
Mother Brain - Main Trading AI
Combines existing LSTM brain with RL and manages child agents
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json
import os
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler

# Import child agents
import sys
sys.path.append(os.path.dirname(__file__))
from child_agents.base_agent import (
    WhaleWatcherAgent, TechnicalAnalystAgent, MarketScannerAgent,
    RugpullDetectorAgent, ReportCoordinatorAgent
)

# Configuration
MOTHER_MODEL_PATH = "R:/Redline_Data/ai_models/mother_brain/mother_v1.pth"
CHECKPOINT_DIR = "R:/Redline_Data/checkpoints/"
EVOLUTION_LOG_DIR = "R:/Redline_Data/evolution_logs/"

# Logger
logger = logging.getLogger("MOTHER_BRAIN")
logging.basicConfig(level=logging.INFO)

# LSTM Network (from existing brain.py)
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class MotherBrain:
    """
    Mother Brain - Main Trading AI
    - Manages child agents
    - Makes final trading decisions
    - Learns via reward/punishment (RL)
    - Can kill weak children and birth new ones
    """
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # LSTM model for price prediction (from existing brain.py)
        self.lstm_model = None
        
        # RL model for decision making (from real_training.py)
        self.rl_model = None
        
        # Child agents
        self.children = {}
        self.next_child_id = 1
        self.generation = 1
        
        # Performance tracking
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.current_balance = 10000.0  # Starting capital
        
        # Load or initialize
        self._initialize()
        
        logger.info("🧠 [Mother Brain] Initialized")
    
    def _initialize(self):
        """Initialize Mother Brain and spawn initial children"""
        # Load LSTM if exists
        self._load_lstm()
        
        # Load RL model if exists
        self._load_rl_model()
        
        # Spawn initial children
        self._spawn_initial_children()
    
    def _load_lstm(self):
        """Load LSTM price prediction model"""
        if os.path.exists(MOTHER_MODEL_PATH):
            try:
                self.lstm_model = CryptoLSTM(hidden_size=100)
                self.lstm_model.load_state_dict(torch.load(MOTHER_MODEL_PATH, map_location=self.device))
                self.lstm_model.eval()
                logger.info("✅ [Mother Brain] LSTM model loaded")
            except Exception as e:
                logger.error(f"❌ [Mother Brain] Failed to load LSTM: {e}")
                self.lstm_model = None
        else:
            logger.warning("⚠️ [Mother Brain] No LSTM model found, will use RL only")
    
    def _load_rl_model(self):
        """Load RL decision-making model"""
        rl_path = os.path.join(CHECKPOINT_DIR, "mother_rl_model.zip")
        if os.path.exists(rl_path):
            try:
                self.rl_model = PPO.load(rl_path)
                logger.info("✅ [Mother Brain] RL model loaded")
            except Exception as e:
                logger.error(f"❌ [Mother Brain] Failed to load RL: {e}")
                self.rl_model = None
        else:
            logger.warning("⚠️ [Mother Brain] No RL model found, will train new one")
    
    def _spawn_initial_children(self):
        """Spawn initial set of child agents"""
        logger.info("👶 [Mother Brain] Spawning initial children...")
        
        # Spawn one of each type
        self.birth_child("whale_watcher", WhaleWatcherAgent)
        self.birth_child("technical_analyst", TechnicalAnalystAgent)
        self.birth_child("market_scanner", MarketScannerAgent)
        self.birth_child("rugpull_detector", RugpullDetectorAgent)
        self.birth_child("report_coordinator", ReportCoordinatorAgent)
        
        logger.info(f"✅ [Mother Brain] {len(self.children)} children spawned")
    
    def birth_child(self, specialty, agent_class, parent_dna=None):
        """
        Birth a new child agent
        Can mutate from parent DNA or start fresh
        """
        child_id = f"{specialty}_{self.next_child_id:03d}"
        self.next_child_id += 1
        
        # Create child
        child = agent_class(
            agent_id=child_id,
            specialty=specialty,
            generation=self.generation
        )
        
        # If parent DNA provided, mutate it
        if parent_dna:
            child.dna = child.mutate()
        
        # Add to children
        self.children[child_id] = child
        
        logger.info(f"👶 [Mother Brain] Born: {child_id} (Gen {self.generation})")
        return child
    
    def kill_child(self, child_id, reason="low_performance"):
        """
        Kill underperforming child
        Log the death for evolution tracking
        """
        if child_id not in self.children:
            return
        
        child = self.children[child_id]
        
        # Log death
        death_log = {
            'timestamp': datetime.now().isoformat(),
            'child_id': child_id,
            'specialty': child.specialty,
            'generation': child.generation,
            'reason': reason,
            'final_accuracy': child.get_accuracy(),
            'total_reports': child.total_reports,
            'contribution': child.contribution_to_profit
        }
        
        # Save to evolution log
        log_file = os.path.join(EVOLUTION_LOG_DIR, "kill_log.json")
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        logs.append(death_log)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        # Remove child
        del self.children[child_id]
        
        logger.info(f"💀 [Mother Brain] Killed: {child_id} - Reason: {reason}")
    
    def collect_reports(self, market_data):
        """
        Collect reports from all children
        """
        reports = []
        
        for child_id, child in self.children.items():
            try:
                if child.specialty == "report_coordinator":
                    # Coordinator gets sibling reports
                    report = child.analyze(reports)
                else:
                    report = child.analyze(market_data)
                reports.append(report)
            except Exception as e:
                logger.error(f"❌ [Mother Brain] Error from {child_id}: {e}")
        
        return reports
    
    def make_decision(self, market_data, child_reports):
        """
        Make final trading decision based on:
        1. LSTM price prediction
        2. Child agent reports
        3. RL policy
        """
        decision = {
            'action': 'HOLD',
            'confidence': 0.5,
            'reasoning': []
        }
        
        # 1. LSTM Prediction
        lstm_signal = None
        if self.lstm_model and isinstance(market_data, pd.DataFrame):
            try:
                predicted_price, confidence, signal = self._lstm_predict(market_data)
                decision['reasoning'].append(f"LSTM: {signal} ({confidence:.2f})")
                lstm_signal = signal
            except Exception as e:
                logger.error(f"❌ [Mother Brain] LSTM error: {e}")
        
        # 2. Aggregate child reports
        child_signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0
        
        for report in child_reports:
            signal = report.get('signal', 'NEUTRAL')
            if signal in child_signals:
                # Weight by child's accuracy
                weight = report.get('accuracy', 0.5)
                child_signals[signal] += weight
                total_confidence += weight
        
        # Normalize
        if total_confidence > 0:
            for sig in child_signals:
                child_signals[sig] /= total_confidence
        
        decision['reasoning'].append(f"Children: BUY={child_signals['BUY']:.2f} SELL={child_signals['SELL']:.2f}")
        
        # 3. Final decision (combine LSTM + children)
        buy_score = child_signals['BUY']
        sell_score = child_signals['SELL']
        
        if lstm_signal == 'BUY':
            buy_score += 0.3
        elif lstm_signal == 'SELL':
            sell_score += 0.3
        
        # Make decision
        if buy_score > 0.6:
            decision['action'] = 'BUY'
            decision['confidence'] = buy_score
        elif sell_score > 0.6:
            decision['action'] = 'SELL'
            decision['confidence'] = sell_score
        else:
            decision['action'] = 'HOLD'
            decision['confidence'] = 0.5
        
        logger.info(f"🧠 [Mother Brain] Decision: {decision['action']} (Conf: {decision['confidence']:.2f})")
        return decision
    
    def _lstm_predict(self, df):
        """LSTM price prediction (from existing brain.py)"""
        SEQ_LENGTH = 60
        
        if len(df) < SEQ_LENGTH:
            return 0, 0, "NEUTRAL"
        
        data = df['close'].tail(SEQ_LENGTH).values.reshape(-1, 1)
        current_price = data[-1][0]
        
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data)
        
        X_input = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction_scaled = self.lstm_model(X_input)
            predicted_price = self.scaler.inverse_transform(prediction_scaled.numpy())[0][0]
        
        diff_percent = ((predicted_price - current_price) / current_price) * 100
        
        signal = "NEUTRAL"
        confidence = 0.5
        threshold = 0.10
        
        if diff_percent > threshold:
            signal = "BUY"
            confidence = min(0.6 + (diff_percent * 2.5), 0.99)
        elif diff_percent < -threshold:
            signal = "SELL"
            confidence = min(0.6 + (abs(diff_percent) * 2.5), 0.99)
        
        return predicted_price, confidence, signal
    
    def learn_from_trade(self, trade_result):
        """
        Learn from trade outcome (reward/punishment)
        """
        profit = trade_result.get('profit', 0)
        was_profitable = profit > 0
        
        # Update Mother's stats
        self.total_trades += 1
        if was_profitable:
            self.profitable_trades += 1
        self.total_profit += profit
        self.current_balance += profit
        
        # Reward/Punishment
        if was_profitable:
            reward = profit * 10  # Amplify reward
            logger.info(f"🍪 [Mother Brain] REWARD: +${profit:.2f} (Total: ${self.total_profit:.2f})")
        else:
            punishment = abs(profit) * 10  # Amplify punishment
            logger.info(f"💔 [Mother Brain] PUNISHMENT: -${abs(profit):.2f}")
        
        # Update children who contributed
        for child_id in trade_result.get('contributing_children', []):
            if child_id in self.children:
                self.children[child_id].update_performance(
                    was_correct=was_profitable,
                    profit_contribution=profit / len(trade_result['contributing_children'])
                )
        
        # Evolution check (every 100 trades)
        if self.total_trades % 100 == 0:
            self.evolve_children()
    
    def evolve_children(self):
        """
        Evolution cycle:
        1. Rank children by performance
        2. Kill bottom 30%
        3. Birth new children with mutations
        """
        logger.info("🧬 [Mother Brain] Starting evolution cycle...")
        
        # Rank children by accuracy
        ranked = sorted(
            self.children.items(),
            key=lambda x: x[1].get_accuracy(),
            reverse=True
        )
        
        # Kill bottom 30%
        kill_count = max(1, len(ranked) // 3)
        for i in range(kill_count):
            child_id, child = ranked[-(i+1)]
            if child.get_accuracy() < 0.5:  # Only kill if truly bad
                self.kill_child(child_id, reason="evolution_culling")
        
        # Birth new children from best performers
        best_children = ranked[:3]
        for _, parent in best_children:
            if len(self.children) < 10:  # Max 10 children
                self.birth_child(
                    specialty=parent.specialty,
                    agent_class=type(parent),
                    parent_dna=parent.dna
                )
        
        self.generation += 1
        logger.info(f"✅ [Mother Brain] Evolution complete - Generation {self.generation}")
    
    def save_checkpoint(self):
        """Save Mother Brain state"""
        checkpoint = {
            'generation': self.generation,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'total_profit': self.total_profit,
            'current_balance': self.current_balance,
            'children_count': len(self.children),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mother_brain_gen{self.generation}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save all children
        for child in self.children.values():
            child.save_checkpoint()
        
        logger.info(f"💾 [Mother Brain] Checkpoint saved (Gen {self.generation})")

# Singleton instance
_mother_brain_instance = None

def get_mother_brain():
    """Get Mother Brain singleton"""
    global _mother_brain_instance
    if _mother_brain_instance is None:
        _mother_brain_instance = MotherBrain()
    return _mother_brain_instance
