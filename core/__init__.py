# core/__init__.py

# FIX: Aktualizacja importów do wersji Security V2
# Teraz eksportujemy HASH i SALT zamiast jawnego kodu.
from .config import THEMES, SECURITY_OVERRIDE_HASH, SECURITY_SALT, REQUIRED_LIBS
from .logger import log_event
from .bootstrap import init_session, session_watchdog
