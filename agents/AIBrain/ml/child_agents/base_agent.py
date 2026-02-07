"""
AIBrain v3.0 - Base Agent with DNA/Evolution System
Merged from Laptop (DNA system) + Desktop (async interface)
"""
import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from abc import ABC, abstractmethod

# R: Drive paths (playground for children)
PLAYGROUND_DIR = "R:/Redline_Data/playground/"
TRAINING_DB_DIR = "R:/Redline_Data/training_db/"
CHECKPOINT_DIR = "R:/Redline_Data/checkpoints/"
EVOLUTION_LOG_DIR = "R:/Redline_Data/evolution_logs/"


class BaseAgent(ABC):
    """
    Base class for all child agents (v3.0 Merged)
    
    Features from Laptop:
    - DNA (hyperparameters) system
    - Performance tracking
    - Checkpoint save/load
    - Mutation for evolution
    
    Features from Desktop:
    - Async analyze() interface
    - get_signal_for_attention() for Mother Brain v3
    """
    
    def __init__(self, name: str, specialty: str = None, generation: int = 1):
        self.name = name
        self.agent_id = f"{name}_{datetime.now().strftime('%H%M%S')}"
        self.specialty = specialty or name
        self.generation = generation
        self.birth_time = datetime.now()
        
        # Performance tracking (from Laptop)
        self.total_reports = 0
        self.correct_signals = 0
        self.false_positives = 0
        self.contribution_to_profit = 0.0
        
        # DNA (hyperparameters) - override in subclass
        self.dna = self._initialize_dna()
        
        # Model (if neural network based)
        self.model = None
        
        # Last analysis result (for Desktop v3.0)
        self.last_analysis = {'signal': 'NEUTRAL', 'score': 0.0}
        
        # Playground path (personal training space on R:)
        self.playground_path = os.path.join(PLAYGROUND_DIR, f"{self.specialty}_{self.agent_id}")
        os.makedirs(self.playground_path, exist_ok=True)
    
    def _initialize_dna(self) -> dict:
        """Initialize agent's DNA (hyperparameters) - override in subclass"""
        return {}
    
    @abstractmethod
    async def analyze(self, market_data: dict) -> dict:
        """
        Analyze market and generate report (async for Desktop v3.0)
        
        Args:
            market_data: dict with 'symbol', 'df', etc.
        
        Returns:
            dict with signal, score, confidence, reasoning
        """
        pass
    
    def get_signal_for_attention(self) -> float:
        """Return normalized score for Attention mechanism (-1 to 1)"""
        return self.last_analysis.get('score', 0.0)
    
    def get_accuracy(self) -> float:
        """Calculate agent's accuracy"""
        if self.total_reports == 0:
            return 0.5
        return self.correct_signals / self.total_reports
    
    def update_performance(self, was_correct: bool, profit_contribution: float = 0.0):
        """Update performance metrics after Mother's trade"""
        self.total_reports += 1
        if was_correct:
            self.correct_signals += 1
        else:
            self.false_positives += 1
        self.contribution_to_profit += profit_contribution
        
        # Save performance to R: drive
        self._save_performance()
    
    def _save_performance(self):
        """Save performance metrics to playground"""
        try:
            perf_file = os.path.join(self.playground_path, "performance.json")
            performance = {
                'agent_id': self.agent_id,
                'name': self.name,
                'specialty': self.specialty,
                'generation': self.generation,
                'birth_time': self.birth_time.isoformat(),
                'total_reports': self.total_reports,
                'correct_signals': self.correct_signals,
                'false_positives': self.false_positives,
                'accuracy': self.get_accuracy(),
                'contribution_to_profit': self.contribution_to_profit,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(perf_file, 'w') as f:
                json.dump(performance, f, indent=2)
        except Exception:
            pass  # Silent fail for performance saving
    
    def save_checkpoint(self):
        """Save agent state to R: drive"""
        try:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"{self.specialty}_{self.agent_id}_gen{self.generation}.pth"
            )
            
            checkpoint = {
                'agent_id': self.agent_id,
                'name': self.name,
                'specialty': self.specialty,
                'generation': self.generation,
                'dna': self.dna,
                'performance': {
                    'accuracy': self.get_accuracy(),
                    'total_reports': self.total_reports,
                    'contribution': self.contribution_to_profit
                },
                'birth_time': self.birth_time.isoformat()
            }
            
            if self.model is not None:
                checkpoint['model_state'] = self.model.state_dict()
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"[{self.name}] Checkpoint save error: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load agent state from R: drive"""
        checkpoint = torch.load(checkpoint_path)
        self.agent_id = checkpoint['agent_id']
        self.generation = checkpoint['generation']
        self.dna = checkpoint['dna']
        
        if 'model_state' in checkpoint and self.model is not None:
            self.model.load_state_dict(checkpoint['model_state'])
        
        print(f"[{self.specialty}] Loaded from checkpoint (Gen {self.generation})")
    
    def mutate(self) -> dict:
        """Mutate DNA (hyperparameters) for evolution"""
        import random
        
        mutated_dna = self.dna.copy()
        
        # Randomly mutate some parameters
        for key in mutated_dna:
            if random.random() < 0.3:  # 30% mutation chance
                if isinstance(mutated_dna[key], float):
                    mutated_dna[key] *= random.uniform(0.8, 1.2)
                elif isinstance(mutated_dna[key], int):
                    mutated_dna[key] = int(mutated_dna[key] * random.uniform(0.8, 1.2))
        
        return mutated_dna
    
    
    def get_report(self) -> dict:
        """Generate standard report format for Mother"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'specialty': self.specialty,
            'timestamp': datetime.now().isoformat(),
            'accuracy': self.get_accuracy(),
            'generation': self.generation,
            **self.last_analysis
        }
    
    def get_status_report(self):
        """Standard short report for Father Brain (LLM)"""
        sig = self.last_analysis.get('signal', 'NEUTRAL')
        score = self.last_analysis.get('score', 0)
        # Pobieramy reasoning, jeśli istnieje
        reason = self.last_analysis.get('reasoning', 'No specifics')
        
        return f"- {self.name.upper()} (Gen {self.generation}): I see {sig} (Conf: {score:.2f}). Context: {reason}."


