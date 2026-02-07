# whaletrades_scraper.py - Scraper for WhaleTrades.io Data Hub
# Pobiera: RSI Heatmap, Money Flow, Sentiment, Bull Market Peak

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not installed. Run: pip install selenium webdriver-manager")

# =============================================================================
# CONFIG
# =============================================================================

BASE_PATH = "R:/Redline_Data/whaletrades"
DATA_HUB_URL = "https://www.whaletrades.io/data-hub"
MAIN_URL = "https://www.whaletrades.io/"

# =============================================================================
# SIMPLE SCRAPER (requests + BeautifulSoup)
# =============================================================================

def scrape_main_page():
    """Scrape main dashboard for liquidations and sentiment"""
    print("\n📊 Scraping WhaleTrades Main Page...")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(MAIN_URL, headers=headers, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find data elements
        data = {
            'timestamp': datetime.now().isoformat(),
            'source': 'whaletrades.io'
        }
        
        # Look for text content
        text_content = soup.get_text()
        
        # Save raw HTML for analysis
        html_path = os.path.join(BASE_PATH, "raw_html", "main_page.html")
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"   ✅ Raw HTML saved to {html_path}")
        print(f"   📄 Page size: {len(response.text):,} bytes")
        
        # Check if page uses JavaScript rendering
        if 'react' in text_content.lower() or 'next' in text_content.lower():
            print("   ⚠️ Page uses JavaScript rendering - need Selenium")
            return None
        
        return data
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# =============================================================================
# SELENIUM SCRAPER (for JavaScript-rendered pages)
# =============================================================================

def scrape_with_selenium():
    """Use Selenium to scrape JavaScript-rendered content"""
    print("\n🌐 Scraping with Selenium...")
    
    if not SELENIUM_AVAILABLE:
        print("   ❌ Selenium not available")
        return None
    
    try:
        # Setup Chrome options
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Use webdriver-manager
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Main page
        print("   📍 Loading main page...")
        driver.get(MAIN_URL)
        time.sleep(5)  # Wait for JS to render
        
        # Get page source
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Save screenshot
        screenshot_path = os.path.join(BASE_PATH, "screenshots", "main_page.png")
        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
        driver.save_screenshot(screenshot_path)
        print(f"   📸 Screenshot saved: {screenshot_path}")
        
        # Try to find data containers
        all_data = {}
        
        # Look for common data patterns
        # Fear & Greed type indicators
        for div in soup.find_all(['div', 'span', 'p']):
            text = div.get_text(strip=True)
            
            # Sentiment indicators
            if 'fear' in text.lower() and 'greed' in text.lower():
                all_data['sentiment_text'] = text[:200]
            
            # Liquidation data
            if 'liquidation' in text.lower():
                all_data['liquidation_text'] = text[:200]
            
            # RSI data
            if 'rsi' in text.lower():
                all_data['rsi_text'] = text[:200]
        
        # Data Hub page
        print("   📍 Loading data hub...")
        driver.get(DATA_HUB_URL)
        time.sleep(5)
        
        # Save screenshot
        screenshot_path = os.path.join(BASE_PATH, "screenshots", "data_hub.png")
        driver.save_screenshot(screenshot_path)
        print(f"   📸 Screenshot saved: {screenshot_path}")
        
        # Get data hub source
        hub_source = driver.page_source
        hub_soup = BeautifulSoup(hub_source, 'html.parser')
        
        # Save HTML
        html_path = os.path.join(BASE_PATH, "raw_html", "data_hub.html")
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(hub_source)
        print(f"   💾 HTML saved: {html_path}")
        
        # Look for tables
        tables = hub_soup.find_all('table')
        print(f"   📊 Found {len(tables)} tables")
        
        for i, table in enumerate(tables[:5]):
            df = pd.read_html(str(table))[0] if table else None
            if df is not None:
                table_path = os.path.join(BASE_PATH, "tables", f"table_{i}.csv")
                os.makedirs(os.path.dirname(table_path), exist_ok=True)
                df.to_csv(table_path, index=False)
                print(f"   ✅ Table {i} saved: {len(df)} rows")
        
        # Look for chart containers (might have data attributes)
        charts = hub_soup.find_all(['canvas', 'svg'])
        print(f"   📈 Found {len(charts)} chart elements")
        
        # Close driver
        driver.quit()
        
        # Save collected data
        if all_data:
            json_path = os.path.join(BASE_PATH, "scraped_data.json")
            with open(json_path, 'w') as f:
                json.dump(all_data, f, indent=2)
            print(f"   💾 Data saved: {json_path}")
        
        return all_data
        
    except Exception as e:
        print(f"   ❌ Selenium error: {e}")
        return None

# =============================================================================
# API DISCOVERY
# =============================================================================

def discover_api():
    """Try to find API endpoints used by the page"""
    print("\n🔍 Discovering API endpoints...")
    
    # Common API patterns for crypto sites
    potential_apis = [
        "https://www.whaletrades.io/api/data",
        "https://www.whaletrades.io/api/sentiment",
        "https://www.whaletrades.io/api/rsi",
        "https://www.whaletrades.io/api/liquidations",
        "https://www.whaletrades.io/api/whales",
        "https://api.whaletrades.io/v1/data",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    for api_url in potential_apis:
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                print(f"   ✅ Found API: {api_url}")
                try:
                    data = response.json()
                    print(f"      Data: {str(data)[:200]}...")
                    return api_url, data
                except:
                    print(f"      Response: {response.text[:100]}...")
            else:
                print(f"   ❌ {api_url}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {api_url}: {str(e)[:50]}")
    
    return None, None

# =============================================================================
# ALTERNATIVE DATA SOURCES
# =============================================================================

def fetch_hyperliquid_data():
    """Fetch data from Hyperliquid (mentioned on whaletrades for whale positions)"""
    print("\n🐋 Fetching Hyperliquid Whale Data...")
    
    # Hyperliquid has a public API
    try:
        # L2 Book
        url = "https://api.hyperliquid.xyz/info"
        headers = {'Content-Type': 'application/json'}
        
        # Get all mids (prices)
        payload = {"type": "allMids"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Hyperliquid mids: {len(data)} assets")
            
            # Save
            path = os.path.join(BASE_PATH, "hyperliquid", "all_mids.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return None

def fetch_coinglass_data():
    """Fetch liquidation data from Coinglass (alternative)"""
    print("\n📉 Fetching Coinglass Data...")
    
    # Coinglass public endpoints
    try:
        url = "https://open-api.coinglass.com/public/v2/liquidation_history"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Coinglass data received")
            
            path = os.path.join(BASE_PATH, "coinglass", "liquidations.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data
        else:
            print(f"   ⚠️ Coinglass API status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🐋 WHALETRADES.IO SCRAPER")
    print("=" * 60)
    print(f"📅 Start: {datetime.now()}")
    print(f"💾 Save path: {BASE_PATH}")
    print("=" * 60)
    
    os.makedirs(BASE_PATH, exist_ok=True)
    
    results = {}
    
    # 1. Try simple scrape
    results['main_page'] = scrape_main_page()
    
    # 2. Discover API
    api_url, api_data = discover_api()
    if api_data:
        results['api'] = api_data
    
    # 3. Try Selenium
    results['selenium'] = scrape_with_selenium()
    
    # 4. Alternative sources
    results['hyperliquid'] = fetch_hyperliquid_data()
    results['coinglass'] = fetch_coinglass_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for name, data in results.items():
        if data:
            print(f"   ✅ {name}: Data collected")
        else:
            print(f"   ❌ {name}: No data")
    
    print("\n" + "=" * 60)
    print(f"✅ SCRAPING COMPLETE!")
    print(f"📂 Data saved to: {BASE_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
