# fetch_all_data.py - Pobiera wszystkie dane dla AI
# Zapisuje na dysk R:\Redline_Data\

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# KONFIGURACJA
# =============================================================================

BASE_PATH = "R:/Redline_Data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

# API Keys (uzupełnij jeśli masz)
CRYPTOPANIC_API_KEY = "3834926bcbda1487af85ac80a57b117efe6520b0"
WHALE_ALERT_API_KEY = os.getenv("WHALE_ALERT_KEY", "")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_KEY", "")

# =============================================================================
# 1. FEAR & GREED INDEX (FREE - no API key needed)
# =============================================================================

def fetch_fear_greed():
    """Pobiera Fear & Greed Index - darmowe API"""
    print("\n📊 Fetching Fear & Greed Index...")
    
    url = "https://api.alternative.me/fng/?limit=365&format=json"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['value'] = df['value'].astype(int)
            
            # Save
            path = os.path.join(BASE_PATH, "sentiment", "fear_greed.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            
            print(f"   ✅ Fear & Greed: {len(df)} days saved")
            return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return None

# =============================================================================
# 2. BINANCE FUNDING RATES (FREE)
# =============================================================================

def fetch_funding_rates():
    """Pobiera funding rates z Binance Futures"""
    print("\n💹 Fetching Funding Rates...")
    
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000"
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df)} funding rates")
                
        except Exception as e:
            print(f"   ❌ {symbol} error: {e}")
        
        time.sleep(0.5)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        path = os.path.join(BASE_PATH, "futures", "funding_rates.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        combined.to_csv(path, index=False)
        print(f"   💾 Saved {len(combined)} total funding rates")
        return combined
    
    return None

# =============================================================================
# 3. BINANCE OPEN INTEREST (FREE)
# =============================================================================

def fetch_open_interest():
    """Pobiera Open Interest z Binance Futures"""
    print("\n📊 Fetching Open Interest...")
    
    all_data = []
    
    for symbol in SYMBOLS:
        # Current OI
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if 'openInterest' in data:
                record = {
                    'symbol': symbol,
                    'openInterest': float(data['openInterest']),
                    'timestamp': datetime.now().isoformat()
                }
                all_data.append(record)
                print(f"   ✅ {symbol}: OI = {float(data['openInterest']):,.2f}")
                
        except Exception as e:
            print(f"   ❌ {symbol} error: {e}")
        
        time.sleep(0.3)
    
    # Historical OI (ostatnie 30 dni)
    for symbol in SYMBOLS:
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1d&limit=30"
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if isinstance(data, list):
                for record in data:
                    all_data.append({
                        'symbol': symbol,
                        'openInterest': float(record.get('sumOpenInterest', 0)),
                        'timestamp': datetime.fromtimestamp(record['timestamp']/1000).isoformat()
                    })
                    
        except Exception as e:
            pass
        
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "futures", "open_interest.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   💾 Saved {len(df)} OI records")
        return df
    
    return None

# =============================================================================
# 4. BINANCE ORDER BOOK DEPTH (FREE)
# =============================================================================

def fetch_order_book():
    """Pobiera order book depth snapshot"""
    print("\n📈 Fetching Order Book Depth...")
    
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=1000"
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if 'bids' in data and 'asks' in data:
                # Calculate depth levels
                bids = [[float(p), float(q)] for p, q in data['bids'][:100]]
                asks = [[float(p), float(q)] for p, q in data['asks'][:100]]
                
                bid_volume = sum([q for p, q in bids])
                ask_volume = sum([q for p, q in asks])
                
                record = {
                    'symbol': symbol,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'bid_ask_ratio': bid_volume / ask_volume if ask_volume > 0 else 0,
                    'best_bid': bids[0][0] if bids else 0,
                    'best_ask': asks[0][0] if asks else 0,
                    'spread': (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0,
                    'timestamp': datetime.now().isoformat()
                }
                all_data.append(record)
                print(f"   ✅ {symbol}: Bid/Ask ratio = {record['bid_ask_ratio']:.2f}")
                
        except Exception as e:
            print(f"   ❌ {symbol} error: {e}")
        
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "orderbook", "depth_snapshot.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   💾 Saved {len(df)} order book snapshots")
        return df
    
    return None

# =============================================================================
# 5. COINGLASS LIQUIDATIONS (FREE - limited)
# =============================================================================

def fetch_liquidations():
    """Pobiera liquidation data (publiczne dane)"""
    print("\n📉 Fetching Liquidation Data...")
    
    # Binance liquidation stream nie jest bezpośrednio dostępny przez REST
    # Używamy publicznych agregatów
    
    all_data = []
    
    for symbol in SYMBOLS:
        # Binance Force Orders (liquidations) - wymagają WebSocket
        # Użyjemy danych agregatowych z dostępnych źródeł
        
        url = f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={symbol}&limit=100"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list):
                    for liq in data:
                        all_data.append({
                            'symbol': symbol,
                            'side': liq.get('side', ''),
                            'price': float(liq.get('price', 0)),
                            'quantity': float(liq.get('origQty', 0)),
                            'time': datetime.fromtimestamp(liq.get('time', 0)/1000).isoformat()
                        })
                    print(f"   ✅ {symbol}: {len(data)} liquidations")
            else:
                print(f"   ⚠️ {symbol}: No public liquidation data")
                
        except Exception as e:
            print(f"   ⚠️ {symbol}: Liquidation data requires auth")
        
        time.sleep(0.5)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "liquidations", "recent_liquidations.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   💾 Saved {len(df)} liquidation records")
        return df
    else:
        print("   ⚠️ Liquidation data requires API access or WebSocket")
    
    return None

# =============================================================================
# 6. CRYPTOPANIC NEWS (API KEY REQUIRED)
# =============================================================================

def fetch_news():
    """Pobiera news z CryptoPanic"""
    print("\n📰 Fetching Crypto News...")
    
    if not CRYPTOPANIC_API_KEY:
        print("   ⚠️ CRYPTOPANIC_TOKEN not set - skipping")
        print("   💡 Get free key at: https://cryptopanic.com/developers/api/")
        return None
    
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&public=true"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'results' in data:
            news = []
            for item in data['results']:
                news.append({
                    'title': item.get('title', ''),
                    'published': item.get('published_at', ''),
                    'source': item.get('source', {}).get('title', ''),
                    'currencies': ','.join([c['code'] for c in item.get('currencies', [])]),
                    'votes_positive': item.get('votes', {}).get('positive', 0),
                    'votes_negative': item.get('votes', {}).get('negative', 0),
                    'url': item.get('url', '')
                })
            
            df = pd.DataFrame(news)
            path = os.path.join(BASE_PATH, "news", "cryptopanic_news.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"   ✅ Saved {len(df)} news articles")
            return df
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return None

# =============================================================================
# 7. WHALE ALERT (API KEY REQUIRED)
# =============================================================================

def fetch_whale_transactions():
    """Pobiera duże transakcje z Binance (FREE) - zamiast płatnego Whale Alert"""
    print("\n🐋 Fetching Large Trades (Free alternative)...")
    
    all_data = []
    
    for symbol in SYMBOLS:
        # Recent large trades from Binance
        url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=1000"
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if isinstance(data, list):
                # Filter large trades (> $50,000)
                price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                price_resp = requests.get(price_url)
                current_price = float(price_resp.json().get('price', 0))
                
                large_trades = []
                for trade in data:
                    qty = float(trade.get('qty', 0))
                    value_usd = qty * current_price
                    
                    if value_usd > 10000:  # > $10k = significant trade
                        large_trades.append({
                            'symbol': symbol,
                            'quantity': qty,
                            'price': float(trade.get('price', 0)),
                            'value_usd': value_usd,
                            'is_buyer_maker': trade.get('isBuyerMaker', False),
                            'time': datetime.fromtimestamp(trade.get('time', 0)/1000).isoformat()
                        })
                
                all_data.extend(large_trades)
                print(f"   ✅ {symbol}: {len(large_trades)} whale trades (>$10k)")
                
        except Exception as e:
            print(f"   ❌ {symbol} error: {e}")
        
        time.sleep(0.5)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "whales", "large_trades.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   💾 Saved {len(df)} whale trades")
        return df
    else:
        print("   ⚠️ No large trades found")
    
    return None

# =============================================================================
# 8. BINANCE TOP TRADER POSITIONS (FREE)
# =============================================================================

def fetch_top_trader_positions():
    """Pobiera pozycje top traderów"""
    print("\n👥 Fetching Top Trader Positions...")
    
    all_data = []
    
    for symbol in SYMBOLS:
        # Long/Short ratio
        url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1d&limit=30"
        
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if isinstance(data, list):
                for record in data:
                    all_data.append({
                        'symbol': symbol,
                        'longShortRatio': float(record.get('longShortRatio', 0)),
                        'longAccount': float(record.get('longAccount', 0)),
                        'shortAccount': float(record.get('shortAccount', 0)),
                        'timestamp': datetime.fromtimestamp(record['timestamp']/1000).isoformat()
                    })
                print(f"   ✅ {symbol}: {len(data)} L/S records")
                
        except Exception as e:
            print(f"   ❌ {symbol} error: {e}")
        
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "futures", "long_short_ratio.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"   💾 Saved {len(df)} L/S ratio records")
        return df
    
    return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🚀 FETCHING ALL AI TRAINING DATA")
    print("=" * 60)
    print(f"📅 Start: {datetime.now()}")
    print(f"💾 Save path: {BASE_PATH}")
    print("=" * 60)
    
    results = {}
    
    # Free APIs (no key needed)
    results['fear_greed'] = fetch_fear_greed()
    results['funding_rates'] = fetch_funding_rates()
    results['open_interest'] = fetch_open_interest()
    results['order_book'] = fetch_order_book()
    results['liquidations'] = fetch_liquidations()
    results['top_traders'] = fetch_top_trader_positions()
    
    # APIs requiring keys
    results['news'] = fetch_news()
    results['whales'] = fetch_whale_transactions()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for name, data in results.items():
        if data is not None:
            print(f"   ✅ {name}: {len(data)} records")
        else:
            print(f"   ❌ {name}: No data")
    
    print("\n" + "=" * 60)
    print("✅ DATA FETCH COMPLETE!")
    print("=" * 60)
    print(f"📂 Data saved to: {BASE_PATH}")
    print(f"📅 Finished: {datetime.now()}")

if __name__ == "__main__":
    main()
