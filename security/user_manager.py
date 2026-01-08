import json
import os
import secrets
import hashlib
from datetime import datetime

class UserManager:
    """
    System Zarządzania Tożsamością (IAM).
    Obsługuje logowanie, hashowanie i system próśb o dostęp.
    """
    
    DB_FILE = os.path.join("assets", "users_db.json")
    
    # Domyślny Root (admin / admin123) - SHA256
    DEFAULT_ROOT = {
        "admin": {
            "hash": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9", 
            "role": "ROOT",
            "contact": "sysadmin",
            "created": "2026-01-01"
        }
    }

    @staticmethod
    def _ensure_assets():
        if not os.path.exists("assets"):
            os.makedirs("assets")

    @staticmethod
    def hash_password(password):
        """Pomocnicza funkcja do hashowania (SHA-256)."""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def load_db():
        """Ładuje bazę z obsługą błędów JSON."""
        UserManager._ensure_assets()
        
        # Inicjalizacja, jeśli plik nie istnieje
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
        """
        Weryfikuje użytkownika i hasło.
        FIX: Hashuje hasło wejściowe przed porównaniem z bazą.
        """
        db = UserManager.load_db()
        
        if user in db['active']:
            stored_data = db['active'][user]
            stored_hash = stored_data.get('hash', '')
            
            # Hashujemy wpisane hasło, aby pasowało do formatu w bazie
            input_hash = UserManager.hash_password(plain_password)
            
            # Bezpieczne porównanie hashów
            if secrets.compare_digest(stored_hash, input_hash):
                return stored_data.get('role', 'VIEWER')
        
        return None

    @staticmethod
    def request_account(username, password, contact):
        """Dodaje prośbę o konto do kolejki 'pending'."""
        db = UserManager.load_db()
        
        if username in db['active']:
            return False, "User already active."
        
        if username in db['pending']:
            return False, "Request pending approval."
            
        db['pending'][username] = {
            "hash": UserManager.hash_password(password),
            "contact": contact,
            "ts": datetime.now().isoformat(),
            "status": "WAITING_FOR_ADMIN"
        }
        
        UserManager.save_db(db)
        return True, "Request submitted to Admin."

    @staticmethod
    def approve_user(user, role="USER"):
        db = UserManager.load_db()
        if user in db['pending']:
            user_data = db['pending'].pop(user)
            user_data['role'] = role
            user_data['allowed_ips'] = [] 
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