"""
AIBrain v2.0 - Brain Recorder (Czarna Skrzynka)
Records all Mother Brain decisions for training data collection
"""
import csv
import os
from datetime import datetime

class BrainRecorder:
    def __init__(self, filepath="data/training_data/mother_decisions.csv"):
        self.filepath = filepath
        self._ensure_directory()
        self._ensure_file_headers()

    def _ensure_directory(self):
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _ensure_file_headers(self):
        headers = [
            'timestamp', 'symbol', 
            'scanner_signal', 'technician_signal', 'whale_signal', 'sentiment_signal',
            'rugpull_signal', 'portfolio_signal',
            'attn_scanner', 'attn_tech', 'attn_whale', 'attn_sent', 'attn_rug', 'attn_port',
            'volatility', 'btc_trend', 'funding_rate',
            'final_action', 'confidence'
        ]
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def record_decision(self, symbol, agent_signals, attention_weights, market_context, final_decision):
        try:
            # Padding wag zerami, jeśli jest ich mniej niż 6
            w = list(attention_weights) + [0]*(6-len(attention_weights))
            
            row = [
                datetime.now().isoformat(),
                symbol,
                agent_signals.get('scanner', 0),
                agent_signals.get('technician', 0),
                agent_signals.get('whale_watcher', 0),
                agent_signals.get('sentiment', 0),
                agent_signals.get('rugpull_detector', 0),
                agent_signals.get('portfolio_manager', 0),
                w[0], w[1], w[2], w[3], w[4], w[5],
                market_context.get('volatility', 0),
                market_context.get('btc_trend', 0),
                market_context.get('funding_rate', 0),
                final_decision['action'],
                final_decision['confidence']
            ]
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return True
        except Exception as e:
            print(f"RECORDER ERROR: {e}")
            return False
