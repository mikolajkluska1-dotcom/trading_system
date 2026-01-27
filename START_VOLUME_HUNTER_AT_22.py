"""
OVERNIGHT LAUNCHER - DAY 2
Starts Volume Hunter training at 22:00
"""
import subprocess
import sys
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LAUNCHER_DAY2")

def wait_until_22():
    """Wait until 22:00"""
    while True:
        now = datetime.now()
        if now.hour >= 22 or now.hour < 6:
            break
        
        wait_minutes = (22 - now.hour) * 60 - now.minute
        logger.info(f"⏰ Waiting {wait_minutes} minutes until 22:00...")
        time.sleep(300)

def main():
    logger.info("=" * 80)
    logger.info("🌙 DAY 2: VOLUME HUNTER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
    
    wait_until_22()
    
    logger.info("\n🚀 STARTING VOLUME HUNTER TRAINING...")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, "agents/AIBrain/ml/train_volume_hunter.py"],
            check=True,
            capture_output=False
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ VOLUME HUNTER TRAINING COMPLETED!")
        logger.info("=" * 80)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
