"""
REDLINE MEGA DATA DOWNLOADER
============================
Pobiera pełne dane historyczne 2022-2024 dla wszystkich modeli AI.
Rozszerzony zestaw symboli + wiele interwałów.
"""
import os
import requests
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
import time

# KONFIGURACJA
BASE_URL = "https://data.binance.vision/data/spot/monthly"
TARGET_DIR = "R:/Redline_Data/bulk_data/klines"

# ROZSZERZONY ZESTAW SYMBOLI
PAIRS = [
    # Tier 1 - Major
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    # Tier 2 - Large Cap
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT",
    # Tier 3 - Mid Cap (więcej zmienności)
    "AVAXUSDT", "MATICUSDT", "LINKUSDT", "ATOMUSDT",
]

# LATA DO POBRANIA
YEARS = [2022, 2023, 2024]

# INTERWAŁY (dla różnych agentów)
INTERVALS = {
    "1h": "Mother Brain + Scanner (główny)",
    "15m": "Technik (szybsze sygnały)",
    "5m": "Scalper (przyszły agent)",
    "1d": "Trend detector (długoterminowy)"
}

def download_and_extract(url, extract_path):
    """Pobiera ZIP i wypakowuje"""
    filename = url.split("/")[-1]
    csv_name = filename.replace(".zip", ".csv")
    local_file = os.path.join(extract_path, csv_name)
    
    if os.path.exists(local_file):
        return "skip"
    
    try:
        r = requests.get(url, timeout=60)
        
        if r.status_code == 404:
            return "404"
        
        if r.status_code != 200:
            return "error"
            
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(extract_path)
        return "ok"
        
    except Exception as e:
        return "error"

def main():
    print("=" * 70)
    print("🔥 REDLINE MEGA DATA DOWNLOADER")
    print("=" * 70)
    print(f"   📁 Target: {TARGET_DIR}")
    print(f"   💰 Symbols: {len(PAIRS)} pairs")
    print(f"   📅 Years: {YEARS}")
    print(f"   ⏱️  Intervals: {list(INTERVALS.keys())}")
    print("=" * 70)
    
    # Generate all tasks
    tasks = []
    for interval in INTERVALS.keys():
        for symbol in PAIRS:
            for year in YEARS:
                for month in range(1, 13):
                    month_str = f"{year}-{month:02d}"
                    path = os.path.join(TARGET_DIR, interval, symbol)
                    os.makedirs(path, exist_ok=True)
                    
                    url = f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{month_str}.zip"
                    tasks.append((url, path, f"{symbol}-{interval}-{month_str}"))
    
    print(f"\n📚 Total: {len(tasks)} files to download")
    print("-" * 70)
    
    stats = {"ok": 0, "skip": 0, "404": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for url, path, name in tasks:
            future = executor.submit(download_and_extract, url, path)
            futures.append((future, name))
            time.sleep(0.02)
        
        for i, (future, name) in enumerate(futures):
            result = future.result()
            stats[result] = stats.get(result, 0) + 1
            
            if result == "ok":
                print(f"✅ [{i+1}/{len(tasks)}] {name}")
            elif result == "skip":
                pass  # Silent skip
            elif result == "404":
                print(f"⚠️  [{i+1}/{len(tasks)}] Not found: {name}")
            else:
                print(f"❌ [{i+1}/{len(tasks)}] Error: {name}")
            
            # Progress every 50
            if (i + 1) % 100 == 0:
                print(f"   📊 Progress: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.0f}%)")
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"   ✅ Downloaded: {stats['ok']}")
    print(f"   ⏩ Skipped:    {stats['skip']}")
    print(f"   ⚠️  Not found:  {stats['404']}")
    print(f"   ❌ Errors:     {stats['error']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
