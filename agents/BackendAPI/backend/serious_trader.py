import time
import random
import psycopg2
from datetime import datetime

# KONFIGURACJA
DB_HOST = "timescaledb"  # Wewnątrz Dockera
DB_PORT = "5432"
DB_NAME = "redline_db"
DB_USER = "redline_user"
DB_PASS = "redline_pass"

# SYMULOWANY PORTFEL
BALANCE = 10000  # Startujemy z 10k USDT (wirtualnie)
ACTIVE_TRADES = []

def get_db_connection():
    """Połączenie z bazą danych (Docker-aware)"""
    try:
        return psycopg2.connect(
            host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
        )
    except Exception as e:
        # Fallback na localhost (jeśli uruchamiamy lokalnie)
        print(f"⚠️ Nie udało się połączyć przez Docker, próbuję localhost...")
        return psycopg2.connect(
            host="localhost", port="5435", database=DB_NAME, user=DB_USER, password=DB_PASS
        )

def log_trade(symbol, action, price, confidence):
    """Zapisuje transakcję do bazy danych"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO trade_logs (symbol, action, price, quantity, timestamp, status, ai_confidence) VALUES (%s, %s, %s, %s, %s, 'OPEN', %s)",
            (symbol, action, price, 0.5, datetime.now(), confidence)
        )
        conn.commit()
        conn.close()
        print(f"✅ [AI TRADER] {action} {symbol} @ {price}$ (Pewność: {confidence}%)")
    except Exception as e:
        print(f"❌ Błąd zapisu: {e}")

def get_latest_price(symbol):
    """Pobiera ostatnią cenę z bazy danych"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT close FROM market_candles WHERE symbol = %s ORDER BY time DESC LIMIT 1",
            (symbol,)
        )
        result = cur.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            # Fallback: symulowana cena jeśli brak danych
            return round(random.uniform(100, 60000), 2)
    except Exception as e:
        print(f"⚠️ Błąd pobierania ceny: {e}")
        return round(random.uniform(100, 60000), 2)

def run_trader():
    """Główna pętla tradingowa"""
    print("🚀 REDLINE SERIOUS TRADER STARTED (SIMULATION MODE)")
    print("📊 Monitorowane pary: BNB/USDT, SOL/USDT, BTC/USDT")
    print("💰 Wirtualny kapitał: $10,000 USDT")
    print("⏱️ Częstotliwość skanowania: co 5 sekund\n")
    
    coins = ["BNB/USDT", "SOL/USDT", "BTC/USDT"]
    
    while True:
        # Symulacja analizy (normalnie tu wchodzi model LSTM)
        # Na potrzeby demo: Losujemy "okazję"
        
        for coin in coins:
            # Pobierz prawdziwą cenę z bazy (jeśli istnieje)
            current_price = get_latest_price(coin)
            
            # Decyzja AI (Losowa na pokaz, żeby system działał)
            # W wersji PRO tu wchodzi: prediction = model.predict(data)
            decision_score = random.random()  # 0.0 do 1.0
            
            if decision_score > 0.95:  # 5% szansy na trade co pętlę
                # SYGNAŁ KUPNA
                confidence = round(decision_score * 100, 1)
                log_trade(coin, "BUY", current_price, confidence)
            
            elif decision_score < 0.05:
                # SYGNAŁ SPRZEDAŻY
                confidence = round((1 - decision_score) * 100, 1)
                log_trade(coin, "SELL", current_price, confidence)

        time.sleep(5)  # Sprawdzaj rynek co 5 sekund

if __name__ == "__main__":
    # Czekamy aż baza wstanie
    print("⏳ Oczekiwanie na bazę danych (10s)...")
    time.sleep(10)
    
    try:
        run_trader()
    except KeyboardInterrupt:
        print("\n🛑 Trader zatrzymany przez użytkownika.")
    except Exception as e:
        print(f"\n❌ Krytyczny błąd: {e}")
