"""
OVERNIGHT TURBO TRAINING LAUNCHER
Starts at 22:00, runs 1000 epochs, finishes by morning
"""
import subprocess
import sys
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LAUNCHER")

def wait_until_22():
    """Wait until 22:00"""
    while True:
        now = datetime.now()
        if now.hour >= 22 or now.hour < 6:  # 22:00 - 6:00 window
            break
        
        wait_minutes = (22 - now.hour) * 60 - now.minute
        logger.info(f"⏰ Waiting {wait_minutes} minutes until 22:00...")
        time.sleep(300)  # Check every 5 minutes

def main():
    logger.info("=" * 80)
    logger.info("🌙 OVERNIGHT TURBO TRAINING LAUNCHER")
    logger.info("=" * 80)
    logger.info(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Wait until 22:00
    wait_until_22()
    
    logger.info("\n🚀 STARTING TURBO TRAINING...")
    logger.info("=" * 80)
    
    # Launch training
    try:
        result = subprocess.run(
            [sys.executable, "agents/AIBrain/ml/turbo_training.py"],
            check=True,
            capture_output=False  # Show output in real-time
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
