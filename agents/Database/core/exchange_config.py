"""
Binance Exchange Configuration Helper
Automatycznie konfiguruje CCXT dla Testnet lub Production
"""
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

def get_binance_exchange(auth=False):
    """
    Tworzy instancję Binance exchange z automatyczną detekcją Testnet
    
    Args:
        auth (bool): Czy użyć API keys z .env (dla handlu)
    
    Returns:
        ccxt.binance: Skonfigurowana instancja exchange
    """
    use_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    }
    
    # Dodajemy klucze API jeśli wymagane
    if auth:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if api_key and api_secret and api_key != "your_key_here":
            config['apiKey'] = api_key
            config['secret'] = api_secret
    
    # Tworzymy exchange
    exchange = ccxt.binance(config)
    
    # Jeśli Testnet, zmieniamy URLs
    if use_testnet:
        exchange.set_sandbox_mode(True)
        print("🔄 Binance TESTNET Mode Enabled (Virtual Money)")
    else:
        print("⚠️ Binance PRODUCTION Mode (Real Money!)")
    
    return exchange


# Przykładowe użycie:
if __name__ == "__main__":
    # Bez autentykacji (publiczne dane)
    exchange_public = get_binance_exchange(auth=False)
    print(f"Public Exchange: {exchange_public.describe()['name']}")
    
    # Z autentykacją (do handlu)
    exchange_private = get_binance_exchange(auth=True)
    print(f"Authenticated Exchange: {exchange_private.describe()['name']}")
    
    # Test połączenia
    try:
        ticker = exchange_public.fetch_ticker('BTC/USDT')
        print(f"✅ BTC Price: ${ticker['last']}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
