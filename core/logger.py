import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# =====================================================
# SYSTEM LOG CONFIGURATION
# =====================================================

LOG_DIR = "assets"
LOG_FILE = os.path.join(LOG_DIR, "system.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Custom success level
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

logging.Logger.success = success

# Configure Logger
logger = logging.getLogger("redline")
logger.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# File Handler (5MB limit, 3 backups)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# Console Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

# Add Handlers
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def log_event(message: str, level: str = "INFO"):
    """
    Standardize message and log it.
    Maintains compatibility with existing log_event(msg, level) calls.
    """
    clean_msg = str(message).replace("\n", " ").strip()[:500]

    level = level.upper()
    if level == "INFO": logger.info(clean_msg)
    elif level == "WARN" or level == "WARNING": logger.warning(clean_msg)
    elif level == "ERROR": logger.error(clean_msg)
    elif level == "SUCCESS" or level == "SUCC": logger.success(clean_msg)
    elif level == "TRADE": logger.info(f"💰 [TRADE] {clean_msg}")
    elif level == "SEC": logger.warning(f"🛡️ [SECURITY] {clean_msg}")
    else: logger.debug(clean_msg)

def get_latest_logs(n: int = 100):
    """Odczytuje ostatnie N linii z pliku logów."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            return lines[-n:]
    except Exception:
        return ["Error reading logs."]
