"""
Master Training Script
Runs all data fetchers and prepares training data for all child agents
"""
import subprocess
import sys
from datetime import datetime

FETCHERS = [
    ("Whale Watcher", "ml/data_fetchers/fetch_whale_data.py"),
    ("Technical Analyst", "ml/data_fetchers/fetch_technical_data.py"),
    ("Market Scanner", "ml/data_fetchers/fetch_scanner_data.py"),
    ("Rugpull Detector", "ml/data_fetchers/fetch_rugpull_data.py"),
]

def run_fetcher(name, script_path):
    """Run a data fetcher script"""
    print(f"\n{'='*60}")
    print(f"Running {name} Data Fetcher")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        if result.returncode == 0:
            print(f"✅ {name} data fetcher completed successfully")
        else:
            print(f"❌ {name} data fetcher failed with code {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ {name} data fetcher timed out")
        return False
    except Exception as e:
        print(f"❌ Error running {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("MASTER TRAINING DATA PREPARATION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    for name, script_path in FETCHERS:
        success = run_fetcher(name, script_path)
        results[name] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    total_success = sum(results.values())
    print(f"\nCompleted: {total_success}/{len(FETCHERS)} fetchers")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 Training data saved to: R:/Redline_Data/training/")
    print("🎯 Ready to train child agents!")

if __name__ == "__main__":
    main()
