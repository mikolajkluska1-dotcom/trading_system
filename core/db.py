import sqlite3
import os
import threading

class Database:
    """
    Centralny moduł bazy danych (SQLite).
    Zapewnia bezpieczeństwo wątkowe dla Orchestratora i API.
    """
    DB_PATH = os.path.join("assets", "trading_system.db")
    _lock = threading.Lock()

    @staticmethod
    def get_connection():
        """Zwraca połączenie z bazą. Wymaga zamknięcia po operacji."""
        if not os.path.exists("assets"):
            os.makedirs("assets")

        conn = sqlite3.connect(Database.DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Pozwala na dostęp po nazwach kolumn
        return conn

    @staticmethod
    def initialize():
        """Tworzy schemat bazy danych jeśli nie istnieje."""
        with Database._lock:
            conn = Database.get_connection()
            cursor = conn.cursor()

            # 1. Tabela Użytkowników
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    role TEXT DEFAULT 'VIEWER',
                    contact TEXT,
                    risk_limit REAL DEFAULT 1000.0,
                    trading_enabled BOOLEAN DEFAULT 0,
                    api_key TEXT,
                    api_secret TEXT,
                    exchange TEXT DEFAULT 'BINANCE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            """)

            # 2. Tabela Oczekiwań (Pending Users)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_users (
                    username TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    contact TEXT,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 3. Tabela Portfela (Assets)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wallet (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL DEFAULT 10000.0,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 4. Tabela Aktywów w Portfelu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    symbol TEXT PRIMARY KEY,
                    amount REAL DEFAULT 0.0,
                    entry_price REAL DEFAULT 0.0,
                    last_price REAL DEFAULT 0.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 5. Historia Transakcji
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    cost REAL NOT NULL,
                    status TEXT,
                    signal_id TEXT,
                    reason TEXT,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Inicjalizacja portfela jeśli pusty
            cursor.execute("SELECT COUNT(*) FROM wallet")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO wallet (balance) VALUES (10000.0)")

            conn.commit()
            conn.close()

# Inicjalizacja przy imporcie
Database.initialize()
