import json
import os
import secrets
import hashlib
from datetime import datetime

class UserManager:
    """
    System Zarządzania Tożsamością (IAM) - GEN 2.5 (API Keys Support)
    Obsługuje: Auth, Roles, Risk Limits, Exchange Credentials
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
    def _ensure_assets():
        if not os.path.exists("assets"):
            os.makedirs("assets")

    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def load_db():
        UserManager._ensure_assets()
        if not os.path.exists(UserManager.DB_FILE):
            db = {"active": UserManager.DEFAULT_ROOT.copy(), "pending": {}}
            UserManager.save_db(db)
            return db     
        try:
            with open(UserManager.DB_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"active": UserManager.DEFAULT_ROOT.copy(), "pending": {}}

    @staticmethod
    def save_db(db):
        UserManager._ensure_assets()
        with open(UserManager.DB_FILE, "w") as f:
            json.dump(db, f, indent=4)

    @staticmethod
    def verify_login(user, plain_password):
        db = UserManager.load_db()
        if user in db['active']:
            stored_data = db['active'][user]
            input_hash = UserManager.hash_password(plain_password)
            if secrets.compare_digest(stored_data.get('hash', ''), input_hash):
                return stored_data.get('role', 'VIEWER')
        return None

    @staticmethod
    def request_account(username, password, contact):
        db = UserManager.load_db()
        if username in db['active']: return False, "User already active."
        if username in db['pending']: return False, "Request pending approval."
            
        db['pending'][username] = {
            "hash": UserManager.hash_password(password),
            "contact": contact,
            "ts": datetime.now().isoformat(),
            "status": "WAITING_FOR_ADMIN"
        }
        UserManager.save_db(db)
        return True, "Request submitted."

    @staticmethod
    def approve_user(user, role="INVESTOR"):
        db = UserManager.load_db()
        if user in db['pending']:
            user_data = db['pending'].pop(user)
            user_data.update({
                "role": role,
                "created_at": datetime.now().isoformat(),
                "trading_enabled": False, 
                "risk_limit": 1000.0,
                # Puste klucze na start
                "exchange_config": {
                    "exchange": "BINANCE",
                    "api_key": "",
                    "api_secret": ""
                },
                "notes": "Approved by Admin"
            })
            db['active'][user] = user_data
            UserManager.save_db(db)
            return True
        return False
        
    @staticmethod
    def reject_user(user):
        db = UserManager.load_db()
        if user in db['pending']:
            del db['pending'][user]
            UserManager.save_db(db)
            return True
        return False

    @staticmethod
    def update_user_settings(username, updates: dict):
        """Edycja ustawień usera - teraz obsługuje klucze API"""
        db = UserManager.load_db()
        if username in db['active']:
            allowed_direct = ['role', 'trading_enabled', 'risk_limit', 'notes', 'contact']
            
            # Aktualizacja pól prostych
            for field in allowed_direct:
                if field in updates:
                    db['active'][username][field] = updates[field]
            
            # Aktualizacja kluczy API (jeśli podano)
            if 'api_key' in updates and 'api_secret' in updates:
                # Jeśli klucze nie są puste, aktualizujemy
                if updates['api_key'] and updates['api_secret']:
                    db['active'][username]['exchange_config'] = {
                        "exchange": "BINANCE", 
                        "api_key": updates['api_key'],
                        "api_secret": updates['api_secret']
                    }
                
            UserManager.save_db(db)
            return True
        return False