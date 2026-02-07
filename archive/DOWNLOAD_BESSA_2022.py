"""
REDLINE BESSA DOWNLOADER - Rok 2022 (The Crash)
===============================================
Pobiera dane z bessy 2022 dla Terapii Szokowej Mother Brain.
Cel: Oduczyć AI strategii "HOLD Forever"
"""
import os
import requests
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
import time

# KONFIGURACJA BESSY
BASE_URL = "https://data.binance.vision/data/spot/monthly"
TARGET_DIR = "R:/Redline_Data/bulk_data/klines/1h"

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
YEAR = 2022  # ROK BESSY
MONTHS = [f"{YEAR}-{m:02d}" for m in range(1, 13)]  # 2022-01 do 2022-12

def download_and_extract(url, extract_path):
    """Pobiera ZIP i wypakowuje"""
    filename = url.split("/")[-1]
    csv_name = filename.replace(".zip", ".csv")
    local_file = os.path.join(extract_path, csv_name)
    
    if os.path.exists(local_file):
        print(f"⏩ Istnieje: {csv_name}")
        return True
    
    try:
        print(f"⬇️  Pobieranie: {filename}")
        r = requests.get(url, timeout=60)
        
        if r.status_code == 404:
            print(f"⚠️  Brak: {filename}")
            return False
        
        if r.status_code != 200:
            print(f"❌ HTTP {r.status_code}: {filename}")
            return False
            
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(extract_path)
        print(f"✅ OK: {csv_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🔥 REDLINE BESSA DOWNLOADER - ROK 2022")
    print("=" * 60)
    print(f"   Cel: {TARGET_DIR}")
    print(f"   Pary: {PAIRS}")
    print(f"   Miesiące: {MONTHS[0]} - {MONTHS[-1]}")
    print("=" * 60)
    
    tasks = []
    
    for symbol in PAIRS:
        path = os.path.join(TARGET_DIR, symbol)
        os.makedirs(path, exist_ok=True)
        
        for month in MONTHS:
            # Format: BTCUSDT-1h-2022-01.zip
            url = f"{BASE_URL}/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
            tasks.append((url, path))
    
    print(f"\n📚 {len(tasks)} plików do pobrania")
    print("-" * 60)
    
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for url, path in tasks:
            future = executor.submit(download_and_extract, url, path)
            futures.append(future)
            time.sleep(0.05)
        
        for f in futures:
            if f.result():
                success += 1
            else:
                failed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ POBRANO: {success} plików")
    print(f"❌ BŁĘDY: {failed} plików")
    print("=" * 60)

if __name__ == "__main__":
    main()
