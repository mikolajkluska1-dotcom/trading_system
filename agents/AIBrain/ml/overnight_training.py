"""
Overnight Training Master Script
Trains Technical Analyst, Volume Hunter, and Mother Brain sequentially
"""
import subprocess
import sys
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('R:/Redline_Data/logs/overnight_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TRAINING_SCRIPTS = [
    {
        'name': 'Technical Analyst',
        'script': 'ml/train_technical_analyst.py',
        'estimated_hours': 3
    }
]

def run_training(script_info):
    """Run a training script"""
    name = script_info['name']
    script = script_info['script']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: {name}")
    logger.info(f"Estimated time: {script_info['estimated_hours']}h")
    logger.info(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            logger.info(f"[{name}] {line.strip()}")
        
        return_code = process.wait()
        elapsed = (time.time() - start_time) / 3600
        
        if return_code == 0:
            logger.info(f"\n✅ {name} completed in {elapsed:.2f}h")
            return True
        else:
            stderr = process.stderr.read()
            logger.error(f"\n❌ {name} failed: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"\n❌ {name} error: {e}")
        return False

def main():
    logger.info("\n" + "="*80)
    logger.info("REDLINE AI - OVERNIGHT TRAINING SESSION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")
    
    results = {}
    overall_start = time.time()
    
    for script in TRAINING_SCRIPTS:
        success = run_training(script)
        results[script['name']] = success
    
    elapsed_total = (time.time() - overall_start) / 3600
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    successful = sum(results.values())
    logger.info(f"\nCompleted: {successful}/{len(TRAINING_SCRIPTS)} agents")
    logger.info(f"Total time: {elapsed_total:.2f}h")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create completion marker
    with open('R:/Redline_Data/training_complete.txt', 'w') as f:
        f.write(f"Training completed at: {datetime.now().isoformat()}\n")
        f.write(f"Successful: {successful}/{len(TRAINING_SCRIPTS)}\n")
        f.write(f"Total time: {elapsed_total:.2f}h\n")
    
    logger.info("\n🎯 AI agents trained and ready!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        raise
