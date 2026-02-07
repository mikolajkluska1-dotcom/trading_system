"""
REDLINE DATA HARVESTER V3 (BULK DOWNLOADER)
===========================================
Pobiera historyczne dane bezpośrednio z archiwum Binance (data.binance.vision).
Jest to 100x szybsze niż przez API i pozwala pobrać dane Tick-by-Tick.

CO POBIERA:
1. trades (tick-by-tick) - dla Mother Brain
2. klines (świece 1m, 5m, 1h, 1d) - dla Scannera
3. aggTrades - dla analizy Wielorybów

LOKALIZACJA: R:/Redline_Data/bulk_data/
"""

import os
import requests
import zipfile
import io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time

# KONFIGURACJA
# ==============================================================================
BASE_URL = "https://data.binance.vision/data/spot/monthly"
TARGET_DIR = "R:/Redline_Data/bulk_data"

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"] # Główne pary (usuń slash)
YEARS = 1  # Ostatni rok (to i tak będzie kilkadziesiąt GB)

# Co pobierać?
TYPES = [
    "klines",      # Świece
    "trades",      # Wszystkie transakcje (BARDZO DUŻE)
    "aggTrades"    # Agregowane (Wieloryby)
]
INTERVALS = ["1m", "5m", "1h", "1d"] # Tylko dla klines
# ==============================================================================

def setup_dirs():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 Utworzono główny katalog: {TARGET_DIR}")

def download_and_extract(url, extract_path):
    """Pobiera ZIP do RAM i wypakowuje na dysk (oszczędza IO)"""
    filename = url.split("/")[-1]
    local_file_path = os.path.join(extract_path, filename.replace(".zip", ".csv"))
    
    if os.path.exists(local_file_path):
        print(f"⏩ Pomijam (istnieje): {filename}")
        return

    try:
        print(f"⬇️  Pobieranie: {filename} ...")
        r = requests.get(url, stream=True)
        if r.status_code == 404:
            print(f"⚠️  Brak pliku na serwerze: {url}")
            return
        
        # Rozpakowanie w locie
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(extract_path)
        print(f"✅ Rozpakowano: {filename}")
        
    except Exception as e:
        print(f"❌ Błąd pobierania {filename}: {e}")

def generate_urls():
    tasks = []
    
    # Generowanie listy miesięcy
    today = datetime.now()
    months = []
    for i in range(YEARS * 12):
        d = today - timedelta(days=30 * i)
        if d > today: continue
        month_str = d.strftime("%Y-%m")
        if month_str not in months:
            months.append(month_str)
    
    # 1. KLINES (Świece)
    if "klines" in TYPES:
        for symbol in PAIRS:
            for interval in INTERVALS:
                path = os.path.join(TARGET_DIR, "klines", interval, symbol)
                if not os.path.exists(path): os.makedirs(path)
                
                for month in months:
                    # Format: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-01.zip
                    url = f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
                    tasks.append((url, path))

    # 2. TRADES (Tick-by-Tick)
    if "trades" in TYPES:
        for symbol in PAIRS:
            path = os.path.join(TARGET_DIR, "trades", symbol)
            if not os.path.exists(path): os.makedirs(path)
            
            for month in months:
                url = f"{BASE_URL}/trades/{symbol}/{symbol}-trades-{month}.zip"
                tasks.append((url, path))

    # 3. AGG TRADES (Wieloryby)
    if "aggTrades" in TYPES:
        for symbol in PAIRS:
            path = os.path.join(TARGET_DIR, "aggTrades", symbol)
            if not os.path.exists(path): os.makedirs(path)
            
            for month in months:
                url = f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{month}.zip"
                tasks.append((url, path))
                
    return tasks

def main():
    setup_dirs()
    print(f"🔥 REDLINE BULK DOWNLOADER V3")
    print(f"   Cel: {TARGET_DIR}")
    print(f"   Pary: {PAIRS}")
    print(f"   Typy: {TYPES}")
    print("---------------------------------------------------")
    
    tasks = generate_urls()
    print(f"📚 Znaleziono {len(tasks)} plików do pobrania.")
    print("   Rozpoczynam pobieranie wielowątkowe (Max 4)...")
    
    # Pobieranie równoległe (szybciej!)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for url, path in tasks:
            executor.submit(download_and_extract, url, path)
            time.sleep(0.1) # Lekki delay dla stabilności

    print("\n✅ POBIERANIE ZAKOŃCZONE!")

if __name__ == "__main__":
    main()
