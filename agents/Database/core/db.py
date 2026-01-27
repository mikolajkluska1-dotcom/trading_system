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
                    notes TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # 1b. Email Settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_settings (
                    id INTEGER PRIMARY KEY,
                    smtp_server TEXT,
                    smtp_port INTEGER,
                    smtp_username TEXT,
                    smtp_password TEXT,
                    from_email TEXT,
                    auto_send_enabled BOOLEAN DEFAULT 0
                )
            """)
            
            # 1c. Email Templates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    subject TEXT,
                    body TEXT,
                    variables TEXT
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

            # 2b. Tabela Pending Registrations (Full Application Data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_registrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    experience TEXT,
                    portfolio_size TEXT,
                    trading_style TEXT,
                    risk_tolerance TEXT,
                    exchanges TEXT,
                    trading_coins TEXT,
                    hear_about TEXT,
                    referral_code TEXT,
                    status TEXT DEFAULT 'pending',
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP,
                    reviewed_by TEXT,
                    admin_notes TEXT
                )
            """)

            # 2c. Audit Log (Admin Actions Tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    admin_username TEXT,
                    action_type TEXT,
                    target TEXT,
                    details TEXT,
                    ip_address TEXT
                )
            """)

            # 2d. Email Log (Email Notifications Tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS email_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    to_email TEXT,
                    subject TEXT,
                    template_name TEXT,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    error_message TEXT
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
