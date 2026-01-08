# core/logger.py
import streamlit as st
from datetime import datetime

# Dozwolone poziomy logowania (Whitelist)
ALLOWED_LEVELS = ["INFO", "WARN", "ERROR", "SEC", "CRITICAL", "TRADE"]

def log_event(message: str, level: str = "INFO"):
    """
    Centralny logger systemowy V2.
    Zabezpieczony przed RAM Exhaustion i Log Injection.
    """
    if "sys" not in st.session_state:
        return

    # 1. Walidacja Levelu (Anti-Spoofing)
    if level not in ALLOWED_LEVELS:
        level = "UNKNOWN"

    # 2. Sanityzacja i Truncate (Anti-DoS)
    # Usuwamy znaki nowej linii i tniemy do 500 znaków
    clean_msg = str(message).replace("\n", " ").strip()[:500]
    
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] [{level}] {clean_msg}"

    # 3. Bezpieczny zapis
    if "logs" in st.session_state["sys"]:
        st.session_state["sys"]["logs"].insert(0, entry)
        
        # Hard Limit pamięci podręcznej (ostatnie 200 zdarzeń)
        if len(st.session_state["sys"]["logs"]) > 200:
            st.session_state["sys"]["logs"].pop()