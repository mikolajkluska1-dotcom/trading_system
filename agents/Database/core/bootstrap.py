# core/bootstrap.py
import streamlit as st
import time

def init_session():
    """
    Inicjalizacja globalnego stanu aplikacji.
    Wersja V2: Idempotentna i bezpieczna dla wyścigu wątków.
    """
    # Domyślne wartości
    defaults = {
        "sys": {
            "auth": False,
            "role": None,
            "user": None,
            "theme": "MATRIX",
            "logs": [],         # Logi zostają nawet po wylogowaniu
            "breach": False,    # Flaga włamania zostaje
            "bank_link": False,
        },
        "last_active": time.time(),
        "initialized": True
    }

    # Bezpieczna inicjalizacja (tylko brakujących kluczy)
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def session_watchdog():
    """
    Strażnik sesji (Watchdog).
    V2: Soft-Reset (zachowuje dowody włamania) + Bezpieczny dostęp.
    """
    TIMEOUT_SECONDS = 300  # 5 minut

    # 1. Bezpieczne sprawdzenie inicjalizacji
    if "last_active" not in st.session_state or "sys" not in st.session_state:
        return

    # 2. Bezpieczny pobór statusu auth (unikamy KeyError)
    is_auth = st.session_state["sys"].get("auth", False)

    if is_auth:
        try:
            # Walidacja typu last_active (ochrona przed prostym spoofingiem typu string)
            last = float(st.session_state["last_active"])
            idle_time = time.time() - last

            if idle_time > TIMEOUT_SECONDS:
                st.warning("SESSION TIMEOUT. SECURE LOCK.")

                # --- SOFT RESET (Forensic preservation) ---
                # Nie robimy clear(), tylko zerujemy dostęp
                st.session_state["sys"]["auth"] = False
                st.session_state["sys"]["user"] = None
                st.session_state["sys"]["role"] = None
                # Logi i flaga 'breach' zostają w pamięci!

                # Wymuszenie przerysowania UI
                st.rerun()
            else:
                # Heartbeat
                st.session_state["last_active"] = time.time()

        except (ValueError, TypeError):
            # Jeśli ktoś manipulował last_active -> natychmiastowy wylot
            st.session_state["sys"]["auth"] = False
            st.rerun()
