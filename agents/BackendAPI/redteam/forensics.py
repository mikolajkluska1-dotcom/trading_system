import gc
import streamlit as st

class MemoryForensics:
    """
    Analiza pamięci procesu (RAM Scraping).
    Sprawdza, czy po wylogowaniu w pamięci nie zostały wrażliwe dane.
    """

    SUSPICIOUS_KEYS = [
        "api_key", "secret", "token", "password", "hash",
        "wallet", "private", "seed", "admin", "root"
    ]

    @staticmethod
    def scan_session_state():
        """Sprawdza obiekt session_state pod kątem wycieków."""
        leaks = []

        # Jeśli jesteśmy wylogowani, 'sys' powinno być czyste lub zresetowane
        if "sys" in st.session_state:
            sys_dump = str(st.session_state["sys"])

            # Szukamy czy user/role nie wiszą w pamięci
            if "'auth': True" in sys_dump:
                leaks.append("CRITICAL: AUTH FLAG PERSISTED")
            if "admin" in sys_dump or "ROOT" in sys_dump:
                leaks.append("WARN: USER DATA PERSISTED IN SYS")

        return leaks

    @staticmethod
    def scan_runtime_objects():
        """
        Głęboki skan obiektów Pythona (Garbage Collector).
        Symuluje atakującego, który ma dostęp do zrzutu pamięci procesu.
        """
        findings = []
        try:
            # Iterujemy po obiektach w pamięci (OSTROŻNIE - to ciężka operacja)
            # Sprawdzamy tylko słowniki, bo tam najczęściej siedzą configi
            for obj in gc.get_objects():
                if isinstance(obj, dict):
                    # Sprawdzamy klucze słownika
                    for k in obj.keys():
                        if isinstance(k, str):
                            if any(s in k.lower() for s in MemoryForensics.SUSPICIOUS_KEYS):
                                # Ignorujemy wewnętrzne zmienne Pythona/Streamlita
                                val_str = str(obj[k])[:50] # Podgląd wartości
                                if "streamlit" not in val_str and "module" not in val_str:
                                    findings.append(f"FOUND TRACE: {k} = {val_str}...")

                if len(findings) > 5: break # Limit wyników dla bezpieczeństwa testu
        except:
            return ["SCAN ERROR (PERMISSION/MEMORY)"]

        return findings
