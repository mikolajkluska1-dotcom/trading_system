import secrets
import hashlib
import os
from datetime import datetime
from agents.Database.core.db import Database

class UserManager:
    """
    System Zarządzania Tożsamością (IAM) - GEN 3.0 (SQLite Powered)
    """

    DB_FILE = os.path.join("assets", "users_db.json")

    # Domyślny Root
    DEFAULT_ROOT = {
        "admin": {
            "hash": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9", # admin123
            "role": "ROOT",
            "contact": "sysadmin",
            "created": "2026-01-01",
            "risk_limit": 1000000.0,
            "exchange_config": {
                "exchange": "BINANCE",
                "api_key": "",
                "api_secret": ""
            },
            "trading_enabled": True
        }
    }

    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def migrate_if_needed():
        """Migracja z JSON do SQL przy pierwszym uruchomieniu."""
        json_file = os.path.join("assets", "users_db.json")
        if not os.path.exists(json_file):
            return

        import json
        try:
            with open(json_file, "r") as f:
                old_db = json.load(f)

            conn = Database.get_connection()
            cursor = conn.cursor()

            # Migracja aktywnych
            for user, data in old_db.get('active', {}).items():
                ex = data.get('exchange_config', {})
                cursor.execute("""
                    INSERT OR IGNORE INTO users
                    (username, hash, role, contact, risk_limit, trading_enabled, api_key, api_secret, exchange, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user, data.get('hash'), data.get('role'), data.get('contact'),
                    data.get('risk_limit', 1000.0), 1 if data.get('trading_enabled') else 0,
                    ex.get('api_key'), ex.get('api_secret'), ex.get('exchange', 'BINANCE'),
                    data.get('notes')
                ))

            # Migracja oczekujących
            for user, data in old_db.get('pending', {}).items():
                cursor.execute("""
                    INSERT OR IGNORE INTO pending_users (username, hash, contact)
                    VALUES (?, ?, ?)
                """, (user, data.get('hash'), data.get('contact')))

            conn.commit()
            conn.close()

            # Zmiana nazwy starego pliku
            os.rename(json_file, json_file + ".bak")
            print(f" ✅ Migrated users to SQLite from {json_file}")
        except Exception as e:
            print(f" ❌ Migration Error: {e}")

    @staticmethod
    def ensure_default_root():
        """Sprawdza czy istnieje ROOT, jeśli nie - tworzy go z DEFAULT_ROOT."""
        conn = Database.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            for username, data in UserManager.DEFAULT_ROOT.items():
                conn.execute("""
                    INSERT INTO users 
                    (username, hash, role, contact, risk_limit, trading_enabled)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    username, data['hash'], data['role'], data['contact'],
                    data['risk_limit'], 1 if data['trading_enabled'] else 0
                ))
            conn.commit()
            print(" ✅ Initialized default admin user.")
        conn.close()

    @staticmethod
    def verify_login(username, plain_password):
        UserManager.migrate_if_needed()
        UserManager.ensure_default_root()
        conn = Database.get_connection()
        user = conn.execute("SELECT hash, role FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user:
            import secrets
            input_hash = UserManager.hash_password(plain_password)
            if secrets.compare_digest(user['hash'], input_hash):
                return user['role']
            else:
                print(f" [LOGIN FAILED] Hash Mismatch for {username}. DB: {user['hash']} vs Input: {input_hash}")
        else:
             print(f" [LOGIN FAILED] User {username} not found in DB.")
        return None

    @staticmethod
    def request_account(username, password, contact):
        conn = Database.get_connection()
        # Check if already exists
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            conn.close()
            return False, "User already active."
        if conn.execute("SELECT 1 FROM pending_users WHERE username = ?", (username,)).fetchone():
            conn.close()
            return False, "Request pending approval."

        conn.execute("INSERT INTO pending_users (username, hash, contact) VALUES (?, ?, ?)",
                     (username, UserManager.hash_password(password), contact))
        conn.commit()
        conn.close()
        return True, "Request submitted."

    @staticmethod
    def approve_user(username, role="INVESTOR"):
        conn = Database.get_connection()
        pending = conn.execute("SELECT * FROM pending_users WHERE username = ?", (username,)).fetchone()

        if pending:
            conn.execute("""
                INSERT INTO users (username, hash, role, contact, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (username, pending['hash'], role, pending['contact'], "Approved by Admin"))
            conn.execute("DELETE FROM pending_users WHERE username = ?", (username,))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False

    @staticmethod
    def reject_user(username):
        conn = Database.get_connection()
        res = conn.execute("DELETE FROM pending_users WHERE username = ?", (username,)).rowcount
        conn.commit()
        conn.close()
        return res > 0

    @staticmethod
    def load_db():
        """Kompatybilność z FastAPI endpointem dla admina"""
        UserManager.migrate_if_needed()
        conn = Database.get_connection()

        from agents.BackendAPI.security.vault import Vault
        active = {}
        users = conn.execute("SELECT * FROM users").fetchall()
        for u in users:
            d = dict(u)
            # Deszyfracja kluczy dla UI/Engine
            d['api_key'] = Vault.decrypt_string(d.get('api_key', ''))
            d['api_secret'] = Vault.decrypt_string(d.get('api_secret', ''))
            active[u['username']] = d

        pending = {}
        p_users = conn.execute("SELECT * FROM pending_users").fetchall()
        for u in p_users:
            pending[u['username']] = dict(u)

        conn.close()
        return {"active": active, "pending": pending}

    @staticmethod
    def update_user_settings(username, updates: dict):
        conn = Database.get_connection()
        allowed = ['role', 'trading_enabled', 'risk_limit', 'notes', 'contact', 'api_key', 'api_secret', 'exchange']

        from agents.BackendAPI.security.vault import Vault
        fields = []
        values = []
        for k, v in updates.items():
            if k in allowed:
                fields.append(f"{k} = ?")
                val = v
                if k in ['api_key', 'api_secret'] and v:
                    val = Vault.encrypt_string(v)
                values.append(val if k != 'trading_enabled' else (1 if v else 0))

        if not fields:
            conn.close()
            return False

        values.append(username)
        query = f"UPDATE users SET {', '.join(fields)} WHERE username = ?"
        res = conn.execute(query, tuple(values)).rowcount
        conn.commit()
        conn.close()
        return res > 0
