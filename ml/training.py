import os
import time
import json
import pandas as pd
import torch
from datetime import datetime
from core.logger import log_event
from ml.brain import DeepBrain
from ml.knowledge import KnowledgeBase
from data.feed import DataFeed

class OfflineTrainer:
    def __init__(self):
        self.brain = DeepBrain()
        self.is_running = False
        self.report_path = os.path.join("assets", "training_report.json")

    def run_training_session(self, assets, timeframe="1h", epochs=50):
        self.is_running = True
        log_event("TRAINING SEQUENCE INITIATED (V4.5 MLOps)", "INFO")

        start_time = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_version": "DeepBrain V4.5",
            "timeframe": timeframe,
            "epochs": epochs,
            "device": str(self.brain.device),
            "assets": [],
            "errors": [],
            "duration_sec": None
        }

        try:
            log_event("LOADING KNOWLEDGE BASE MEMORY...", "INFO")
            kb_df = KnowledgeBase.load_training_data("system_auto")
            if not kb_df.empty:
                log_event(f"FOUND {len(kb_df)} PATTERNS IN KNOWLEDGE BASE", "INFO")
            else:
                log_event("KNOWLEDGE BASE EMPTY - SKIPPING RL STAGE", "WARN")
        except Exception as e:
            log_event(f"KNOWLEDGE LOAD ERROR: {str(e)}", "ERROR")

        for symbol in assets:
            if not self.is_running:
                log_event("TRAINING INTERRUPTED", "WARN")
                break

            log_event(f"FETCHING MARKET DATA: {symbol} [{timeframe}]", "INFO")

            try:
                df = DataFeed.get_market_data(symbol, timeframe, limit=1000)

                if df is None or df.empty or len(df) < 200:
                    log_event(f"INSUFFICIENT DATA FOR {symbol}", "WARN")
                    results["errors"].append({"asset": symbol, "reason": "INSUFFICIENT_DATA"})
                    continue

                log_event(f"TRAINING MODEL ON {symbol} ({len(df)} rows)...", "INFO")
                success = self.brain.train_on_fly(df, epochs=epochs)

                if success:
                    loss_val = round(float(self.brain.last_train_loss), 6)
                    
                    results["assets"].append({
                        "symbol": symbol,
                        "rows": len(df),
                        "arch": self.brain.current_arch,
                        "loss": loss_val,
                        "status": "SUCCESS"
                    })
                    log_event(f"MODEL UPDATED: {symbol} | Loss: {loss_val} | Arch: {self.brain.current_arch}", "succ")
                else:
                    results["errors"].append({"asset": symbol, "reason": "TRAINING_FAILED"})
                    log_event(f"MODEL TRAINING FAILED FOR {symbol}", "ERROR")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                results["errors"].append({"asset": symbol, "reason": str(e)})
                log_event(f"CRITICAL TRAIN ERROR [{symbol}]: {str(e)}", "ERROR")

        duration = time.time() - start_time
        results["duration_sec"] = round(duration, 2)

        os.makedirs("assets", exist_ok=True)
        try:
            with open(self.report_path, "w") as f:
                json.dump(results, f, indent=4)
            log_event(f"TRAINING REPORT SAVED: {self.report_path}", "INFO")
        except Exception as e:
            log_event(f"FAILED TO SAVE TRAINING REPORT: {str(e)}", "ERROR")

        self.is_running = False
        summary = (
            f"SESSION COMPLETED | Duration: {results['duration_sec']}s | "
            f"Assets: {len(results['assets'])} | Errors: {len(results['errors'])}"
        )
        log_event(summary, "succ")
        return results

if __name__ == "__main__":
    trainer = OfflineTrainer()
    assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    summary = trainer.run_training_session(assets=assets, timeframe="1h", epochs=50)
    print("\n=== OFFLINE TRAINING SUMMARY ===")
    print(json.dumps(summary, indent=4))