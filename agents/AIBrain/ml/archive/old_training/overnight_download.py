"""
OVERNIGHT DATA COLLECTION - MASTER SCRIPT
Downloads ALL training data for Mother Brain and 7 child agents
Total: ~600GB | Time: 12-16 hours
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
        logging.FileHandler('R:/Redline_Data/logs/overnight_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data fetchers in priority order
FETCHERS = [
    {
        'name': 'Technical Analyst',
        'script': 'ml/data_fetchers/fetch_technical_data.py',
        'priority': 1,
        'size_gb': 150,
        'estimated_hours': 10
    },
    {
        'name': 'Market Scanner',
        'script': 'ml/data_fetchers/fetch_scanner_data.py',
        'priority': 2,
        'size_gb': 120,
        'estimated_hours': 7
    },
    {
        'name': 'Whale Watcher',
        'script': 'ml/data_fetchers/fetch_whale_data.py',
        'priority': 3,
        'size_gb': 80,
        'estimated_hours': 10
    },
    {
        'name': 'Rugpull Detector',
        'script': 'ml/data_fetchers/fetch_rugpull_data.py',
        'priority': 4,
        'size_gb': 40,
        'estimated_hours': 5
    }
]

def run_fetcher(fetcher_info):
    """Run a data fetcher and monitor progress"""
    name = fetcher_info['name']
    script = fetcher_info['script']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting: {name}")
    logger.info(f"Expected size: {fetcher_info['size_gb']}GB")
    logger.info(f"Estimated time: {fetcher_info['estimated_hours']}h")
    logger.info(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Run fetcher
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
        
        # Wait for completion
        return_code = process.wait()
        
        elapsed = (time.time() - start_time) / 3600  # hours
        
        if return_code == 0:
            logger.info(f"\n✅ {name} completed successfully in {elapsed:.2f}h")
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
    logger.info("REDLINE AI - OVERNIGHT DATA COLLECTION")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")
    
    logger.info("📊 Download Plan:")
    total_gb = sum(f['size_gb'] for f in FETCHERS)
    total_hours = sum(f['estimated_hours'] for f in FETCHERS)
    logger.info(f"Total data: {total_gb}GB")
    logger.info(f"Estimated time: {total_hours}h (sequential)")
    logger.info(f"With parallelization: ~{total_hours/2:.1f}h\n")
    
    results = {}
    overall_start = time.time()
    
    # Run fetchers sequentially (can be parallelized later)
    for fetcher in FETCHERS:
        success = run_fetcher(fetcher)
        results[fetcher['name']] = success
        
        if not success:
            logger.warning(f"⚠️ {fetcher['name']} failed, continuing with next...")
    
    # Summary
    elapsed_total = (time.time() - overall_start) / 3600
    
    logger.info("\n" + "="*80)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*80)
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    successful = sum(results.values())
    logger.info(f"\nCompleted: {successful}/{len(FETCHERS)} fetchers")
    logger.info(f"Total time: {elapsed_total:.2f}h")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\n🎯 Data saved to: R:/Redline_Data/training_db/")
    logger.info("🧠 Ready for AI training!")
    
    # Create completion marker
    with open('R:/Redline_Data/download_complete.txt', 'w') as f:
        f.write(f"Download completed at: {datetime.now().isoformat()}\n")
        f.write(f"Successful: {successful}/{len(FETCHERS)}\n")
        f.write(f"Total time: {elapsed_total:.2f}h\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Download interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        raise
