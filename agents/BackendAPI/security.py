import os
import time
import hashlib
import random

class HardwareSecurity:
    """
    CORE SECURITY MODULE.
    Odpowiada za symulację uwierzytelniania sprzętowego (USB Key).
    """

    @staticmethod
    def scan_for_key():
        """
        Skanuje porty (symulacja) w poszukiwaniu klucza dostępu.
        Naprawia błąd: AttributeError: type object 'HardwareSecurity' has no attribute 'scan_for_key'
        Zwraca: (bool success, str drive_path)
        """
        # Symulujemy chwilę "pracy" dla efektu UI
        time.sleep(0.8)

        # W trybie PAPER/DEV zawsze zwracamy sukces, żebyś mógł się zalogować
        # W wersji LIVE tutaj byłby kod sprawdzający fizyczne dyski (e.g. win32api.GetLogicalDriveStrings)
        return True, "VIRTUAL_SECURE_DRIVE"

    @staticmethod
    def verify_key_signature(drive_path):
        """
        Symuluje weryfikację kryptograficzną pliku na kluczu.
        """
        time.sleep(0.5)
        # Zawsze True w dev
        return True

    @staticmethod
    def get_hardware_id():
        """Generuje unikalny ID maszyny (fingerprint)"""
        # Symulacja stałego ID dla tej sesji
        return hashlib.sha256(b"REDLINE_DEV_MACHINE").hexdigest()[:16].upper()

    @staticmethod
    def generate_2fa_code():
        """Generuje kod 2FA dla sesji"""
        return f"{random.randint(100, 999)}-{random.randint(100, 999)}"
