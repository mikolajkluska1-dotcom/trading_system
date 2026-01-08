import os
import json
import streamlit as st
from cryptography.fernet import Fernet

class Vault:
    """
    Moduł obronny: Szyfrowanie Portfela (AES-128 via Fernet).
    V3.1: Skonfigurowany pod Trading System (Startowe $10k).
    """
    
    KEY_FILE = os.path.join("assets", "redline.key")
    WALLET_FILE = os.path.join("assets", "wallet.enc")
    DECOY_FILE = os.path.join("assets", "honeypot.txt")

    @staticmethod
    def _ensure_assets_dir():
        if not os.path.exists("assets"):
            os.makedirs("assets")

    @staticmethod
    def _get_cipher():
        """Pobiera klucz. NIGDY nie generuje nowego, jeśli istnieje osierocony portfel."""
        Vault._ensure_assets_dir()
        
        # --- CRITICAL SAFETY CHECK ---
        # Jeśli portfel istnieje, ale klucza brak -> STOP!
        if os.path.exists(Vault.WALLET_FILE) and not os.path.exists(Vault.KEY_FILE):
            st.error("🚨 CRITICAL: ORPHANED WALLET DETECTED. KEY MISSING. ABORTING.")
            return None

        # Generowanie klucza TYLKO jeśli to czysta instalacja
        if not os.path.exists(Vault.KEY_FILE):
            try:
                key = Fernet.generate_key()
                with open(Vault.KEY_FILE, "wb") as f:
                    f.write(key)
            except Exception:
                return None
        
        try:
            with open(Vault.KEY_FILE, "rb") as f:
                key = f.read()
            return Fernet(key)
        except Exception:
            return None

    @staticmethod
    def load_wallet():
        """Bezpieczny odczyt portfela."""
        # 1. Sprawdzenie LOCKDOWN (Red Team simulation)
        if st.session_state.get("sys", {}).get("breach", False):
            return {"balance": 0, "assets": [], "history": [], "LOCKED": True}

        cipher = Vault._get_cipher()
        
        # Jeśli cipher jest None
        if cipher is None:
            return {"balance": 0, "assets": [], "ERROR": "FATAL_KEY_ERROR"}

        # 2. Inicjalizacja (tylko jeśli plik nie istnieje)
        if not os.path.exists(Vault.WALLET_FILE):
            # --- ZMIANA TUTAJ: Dajemy $10,000 na start ---
            data = {
                "balance": 10000.00, 
                "assets": [], 
                "history": [],
                "LOCKED": False # Wymagane przez execution.py
            }
            Vault.save_wallet(data)
            return data
            
        # 3. Próba deszyfracji
        try:
            with open(Vault.WALLET_FILE, "rb") as f:
                encrypted_data = f.read()
            decrypted_json = cipher.decrypt(encrypted_data).decode()
            return json.loads(decrypted_json)
        except Exception:
            return {"balance": 0, "assets": [], "history": [], "ERROR": "DECRYPT_FAIL"}

    @staticmethod
    def save_wallet(data):
        cipher = Vault._get_cipher()
        if cipher:
            try:
                encrypted = cipher.encrypt(json.dumps(data).encode())
                with open(Vault.WALLET_FILE, "wb") as f:
                    f.write(encrypted)
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def deploy_honeypot():
        Vault._ensure_assets_dir()
        try:
            with open(Vault.DECOY_FILE, "w") as f: 
                f.write("root_user: admin\npass: 123456\nseed: apple banana cherry\nBTC_WALLET: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
            return True
        except:
            return False