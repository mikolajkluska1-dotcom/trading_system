import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Konfiguracja połączenia (To są dane z docker-compose.yml)
DB_HOST = "127.0.0.1"
DB_NAME = "redline_db"
DB_USER = "redline_user"
DB_PASS = "redline_pass"
DB_PORT = "5435"

def init_database():
    try:
        # 1. Połączenie z bazą
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        print("✅ Połączono z bazą danych!")

        # 2. Włączenie rozszerzenia TimescaleDB (To robi z niej potwora wydajności)
        print("⏳ Aktywacja TimescaleDB...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # 3. Tworzenie tabeli na świeczki (OHLCV)
        print("🔨 Tworzenie tabeli market_candles...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_candles (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                CONSTRAINT unique_candle UNIQUE (time, symbol)
            );
        """)

        # 4. Zamiana zwykłej tabeli w HIPER-TABELĘ (Magia Timescale)
        # To dzieli dane na kawałki po czasie (chunks), dzięki czemu jest super szybkie
        try:
            cur.execute("SELECT create_hypertable('market_candles', 'time', if_not_exists => TRUE);")
            print("🚀 Tabela zamieniona w HYPERTABLE (Szybki dostęp)!")
        except Exception as e:
            print(f"ℹ️ Hypertable już istnieje lub błąd: {e}")

        # 5. Indeksy dla szybkości (żeby bot nie szukał danych godzinami)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON market_candles (symbol, time DESC);")
        print("⚡ Indeksy utworzone.")

        # 6. Tworzenie tabeli trade_logs (Śledzenie transakcji AI)
        print("🔨 Tworzenie tabeli trade_logs...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                price DECIMAL(18, 8) NOT NULL,
                quantity DECIMAL(18, 8) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
                pnl DECIMAL(18, 8),
                ai_confidence DECIMAL(5, 2)
            );
        """)
        print("✅ Tabela trade_logs utworzona.")
        
        # 7. Indeksy dla trade_logs
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_logs (symbol);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_logs (timestamp DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trade_status ON trade_logs (status);")
        print("⚡ Indeksy dla trade_logs utworzone.")

        cur.close()
        conn.close()
        print("\n🎉 SUKCES! Baza jest gotowa na przyjęcie danych.")

    except Exception as e:
        # Zmiana: wymuszamy wypisanie błędu jako "bezpiecznego" tekstu bez polskich znaków
        print(f"\n❌ BŁĄD: {repr(e)}")
        print("Upewnij się, że Docker działa i port 5432 jest otwarty.")

if __name__ == "__main__":
    init_database()