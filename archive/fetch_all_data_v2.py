# fetch_all_data_v2.py - COMPLETE DATA FETCHER
# All free data sources for AI training

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

CRYPTOPANIC_API_KEY = "3834926bcbda1487af85ac80a57b117efe6520b0"

# =============================================================================
# 1. FEAR & GREED INDEX
# =============================================================================

def fetch_fear_greed():
    print("\n📊 Fetching Fear & Greed Index...")
    url = "https://api.alternative.me/fng/?limit=365&format=json"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'data' in data:
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['value'] = df['value'].astype(int)
            
            path = os.path.join(BASE_PATH, "sentiment", "fear_greed.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"   ✅ Fear & Greed: {len(df)} days")
            return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
    return None

# =============================================================================
# 2. BITCOIN DOMINANCE & GLOBAL MARKET
# =============================================================================

def fetch_global_market():
    print("\n🌍 Fetching Global Market Data...")
    url = "https://api.coingecko.com/api/v3/global"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'data' in data:
            global_data = data['data']
            record = {
                'total_market_cap_usd': global_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume_24h_usd': global_data.get('total_volume', {}).get('usd', 0),
                'btc_dominance': global_data.get('market_cap_percentage', {}).get('btc', 0),
                'eth_dominance': global_data.get('market_cap_percentage', {}).get('eth', 0),
                'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            df = pd.DataFrame([record])
            path = os.path.join(BASE_PATH, "market", "global_market.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            
            print(f"   ✅ BTC Dominance: {record['btc_dominance']:.1f}%")
            print(f"   ✅ Total Market Cap: ${record['total_market_cap_usd']/1e12:.2f}T")
            return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
    return None

# =============================================================================
# 3. TRENDING COINS (Social buzz indicator)
# =============================================================================

def fetch_trending_coins():
    print("\n🔥 Fetching Trending Coins...")
    url = "https://api.coingecko.com/api/v3/search/trending"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'coins' in data:
            trending = []
            for i, coin in enumerate(data['coins'][:10]):
                item = coin.get('item', {})
                trending.append({
                    'rank': i + 1,
                    'name': item.get('name', ''),
                    'symbol': item.get('symbol', ''),
                    'market_cap_rank': item.get('market_cap_rank', 0),
                    'timestamp': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(trending)
            path = os.path.join(BASE_PATH, "sentiment", "trending_coins.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            
            print(f"   ✅ Top trending: {trending[0]['symbol'] if trending else 'N/A'}")
            return df
    except Exception as e:
        print(f"   ❌ Error: {e}")
    return None

# =============================================================================
# 4. BINANCE FUNDING RATES
# =============================================================================

def fetch_funding_rates():
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
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.3)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        path = os.path.join(BASE_PATH, "futures", "funding_rates.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        combined.to_csv(path, index=False)
        return combined
    return None

# =============================================================================
# 5. OPEN INTEREST
# =============================================================================

def fetch_open_interest():
    print("\n📊 Fetching Open Interest...")
    all_data = []
    
    for symbol in SYMBOLS:
        # Current
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            if 'openInterest' in data:
                all_data.append({
                    'symbol': symbol,
                    'openInterest': float(data['openInterest']),
                    'timestamp': datetime.now().isoformat()
                })
                print(f"   ✅ {symbol}: OI = {float(data['openInterest']):,.0f}")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        
        # Historical
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
        except:
            pass
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "futures", "open_interest.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 6. LONG/SHORT RATIO
# =============================================================================

def fetch_long_short():
    print("\n👥 Fetching Long/Short Ratio...")
    all_data = []
    
    for symbol in SYMBOLS:
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
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "futures", "long_short_ratio.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 7. TAKER BUY/SELL RATIO
# =============================================================================

def fetch_taker_ratio():
    print("\n📈 Fetching Taker Buy/Sell Ratio...")
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=1d&limit=30"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            if isinstance(data, list):
                for record in data:
                    all_data.append({
                        'symbol': symbol,
                        'buySellRatio': float(record.get('buySellRatio', 0)),
                        'buyVol': float(record.get('buyVol', 0)),
                        'sellVol': float(record.get('sellVol', 0)),
                        'timestamp': datetime.fromtimestamp(record['timestamp']/1000).isoformat()
                    })
                print(f"   ✅ {symbol}: {len(data)} taker ratios")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "futures", "taker_ratio.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 8. TOP TRADER POSITIONS
# =============================================================================

def fetch_top_trader_positions():
    print("\n🏆 Fetching Top Trader Positions...")
    all_data = []
    
    for symbol in SYMBOLS:
        # Position ratio
        url = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1d&limit=30"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            if isinstance(data, list):
                for record in data:
                    all_data.append({
                        'symbol': symbol,
                        'type': 'position',
                        'longShortRatio': float(record.get('longShortRatio', 0)),
                        'longAccount': float(record.get('longAccount', 0)),
                        'shortAccount': float(record.get('shortAccount', 0)),
                        'timestamp': datetime.fromtimestamp(record['timestamp']/1000).isoformat()
                    })
                print(f"   ✅ {symbol}: {len(data)} position records")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "futures", "top_trader_positions.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 9. ORDER BOOK DEPTH
# =============================================================================

def fetch_order_book():
    print("\n📈 Fetching Order Book Depth...")
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=1000"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if 'bids' in data and 'asks' in data:
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
                    'spread_pct': (asks[0][0] - bids[0][0]) / bids[0][0] * 100 if bids and asks else 0,
                    'timestamp': datetime.now().isoformat()
                }
                all_data.append(record)
                print(f"   ✅ {symbol}: Bid/Ask = {record['bid_ask_ratio']:.2f}")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "orderbook", "depth_snapshot.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 10. LARGE TRADES (Whale detector)
# =============================================================================

def fetch_whale_trades():
    print("\n🐋 Fetching Large Trades...")
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=1000"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if isinstance(data, list):
                price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                price_resp = requests.get(price_url)
                current_price = float(price_resp.json().get('price', 0))
                
                for trade in data:
                    qty = float(trade.get('qty', 0))
                    value_usd = qty * current_price
                    
                    if value_usd > 5000:  # > $5k
                        all_data.append({
                            'symbol': symbol,
                            'quantity': qty,
                            'price': float(trade.get('price', 0)),
                            'value_usd': value_usd,
                            'is_buyer_maker': trade.get('isBuyerMaker', False),
                            'time': datetime.fromtimestamp(trade.get('time', 0)/1000).isoformat()
                        })
                
                whale_count = len([t for t in all_data if t['symbol'] == symbol])
                print(f"   ✅ {symbol}: {whale_count} large trades (>$5k)")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.3)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "whales", "large_trades.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 11. CRYPTO NEWS (CryptoPanic)
# =============================================================================

def fetch_crypto_news():
    print("\n📰 Fetching Crypto News...")
    
    if not CRYPTOPANIC_API_KEY:
        print("   ⚠️ No API key")
        return None
    
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&public=true&filter=hot"
    
    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            
            if 'results' in data:
                news = []
                for item in data['results'][:50]:
                    currencies = item.get('currencies', [])
                    news.append({
                        'title': item.get('title', ''),
                        'published': item.get('published_at', ''),
                        'source': item.get('source', {}).get('title', ''),
                        'currencies': ','.join([c.get('code', '') for c in currencies]) if currencies else '',
                        'domain': item.get('domain', ''),
                        'votes_positive': item.get('votes', {}).get('positive', 0),
                        'votes_negative': item.get('votes', {}).get('negative', 0),
                        'kind': item.get('kind', '')
                    })
                
                df = pd.DataFrame(news)
                path = os.path.join(BASE_PATH, "news", "cryptopanic_news.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df.to_csv(path, index=False)
                print(f"   ✅ News: {len(news)} articles")
                return df
        else:
            print(f"   ⚠️ API returned: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    return None

# =============================================================================
# 12. EXCHANGE INFLOW/OUTFLOW (CryptoQuant alternative)
# =============================================================================

def fetch_exchange_flow():
    print("\n💱 Fetching Exchange Flow Data...")
    
    # Using Binance 24h stats as proxy for exchange activity
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            record = {
                'symbol': symbol,
                'volume_24h': float(data.get('volume', 0)),
                'quote_volume_24h': float(data.get('quoteVolume', 0)),
                'price_change_pct': float(data.get('priceChangePercent', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'trades_count_24h': int(data.get('count', 0)),
                'timestamp': datetime.now().isoformat()
            }
            all_data.append(record)
            print(f"   ✅ {symbol}: Vol ${record['quote_volume_24h']/1e6:.0f}M, {record['trades_count_24h']:,} trades")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.2)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "exchange", "24h_stats.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# 13. HISTORICAL VOLATILITY
# =============================================================================

def fetch_volatility():
    print("\n📉 Calculating Historical Volatility...")
    all_data = []
    
    for symbol in SYMBOLS:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=30"
        try:
            response = requests.get(url, timeout=30)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 5:
                closes = [float(k[4]) for k in data]
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                
                import statistics
                volatility = statistics.stdev(returns) * (365 ** 0.5) * 100  # Annualized %
                
                record = {
                    'symbol': symbol,
                    'volatility_30d': volatility,
                    'avg_daily_return': sum(returns) / len(returns) * 100,
                    'max_daily_return': max(returns) * 100,
                    'min_daily_return': min(returns) * 100,
                    'timestamp': datetime.now().isoformat()
                }
                all_data.append(record)
                print(f"   ✅ {symbol}: Volatility = {volatility:.1f}%")
        except Exception as e:
            print(f"   ❌ {symbol}: {e}")
        time.sleep(0.2)
    
    if all_data:
        df = pd.DataFrame(all_data)
        path = os.path.join(BASE_PATH, "analytics", "volatility.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🚀 COMPLETE DATA FETCHER V2")
    print("=" * 60)
    print(f"📅 Start: {datetime.now()}")
    print(f"💾 Save path: {BASE_PATH}")
    print("=" * 60)
    
    results = {}
    
    # Market Overview
    results['fear_greed'] = fetch_fear_greed()
    results['global_market'] = fetch_global_market()
    results['trending'] = fetch_trending_coins()
    
    # Futures Data
    results['funding_rates'] = fetch_funding_rates()
    results['open_interest'] = fetch_open_interest()
    results['long_short'] = fetch_long_short()
    results['taker_ratio'] = fetch_taker_ratio()
    results['top_traders'] = fetch_top_trader_positions()
    
    # Trading Data
    results['order_book'] = fetch_order_book()
    results['whale_trades'] = fetch_whale_trades()
    results['exchange_stats'] = fetch_exchange_flow()
    results['volatility'] = fetch_volatility()
    
    # News
    results['news'] = fetch_crypto_news()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    success = 0
    failed = 0
    for name, data in results.items():
        if data is not None:
            print(f"   ✅ {name}: {len(data)} records")
            success += 1
        else:
            print(f"   ❌ {name}: No data")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE! {success}/{success+failed} sources fetched")
    print("=" * 60)
    print(f"📂 Data saved to: {BASE_PATH}")

if __name__ == "__main__":
    main()
