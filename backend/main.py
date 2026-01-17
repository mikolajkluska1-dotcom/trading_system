import sys
import os

# --- CRITICAL WINDOWS FIX ---
# Forces UTF-8 encoding to prevent crashes when printing emojis (🧠, 💰)
# This MUST be the first thing the script does.
try:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass
# ----------------------------

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from contextlib import asynccontextmanager

# IMPORTS
from backend.ai_core import RedlineAICore
from backend.events_server import start_background_loop
from core.db import Database

# --- REAL-TIME LOG CAPTURE ---
# This buffer holds the logs to send to the Frontend
LOG_BUFFER: List[str] = []

class MemoryLogHandler(logging.Handler):
    """Intercepts logs and stores the last 50 lines in memory."""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Filter out boring logs (optional)
            if "GET /api" in msg or "WebSocket" in msg:
                return 
            
            LOG_BUFFER.append(msg)
            if len(LOG_BUFFER) > 50:
                LOG_BUFFER.pop(0)
        except Exception:
            self.handleError(record)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
root_logger = logging.getLogger()
memory_handler = MemoryLogHandler()
memory_handler.setFormatter(logging.Formatter('%(message)s')) # Send just the message to UI
root_logger.addHandler(memory_handler)
logger = logging.getLogger("API")

# GLOBAL INSTANCE (The Single Source of Truth)
# This instance is used by both the API and the background loop
ai_core = RedlineAICore(mode="PAPER", timeframe="1h")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("🚀 SYSTEM STARTUP INITIATED")
    Database.initialize()
    
    # Start the background loop thread
    loop_task = asyncio.create_task(start_background_loop(ai_core))
    
    yield
    
    # SHUTDOWN
    logger.info("🛑 SHUTDOWN SIGNAL RECEIVED")
    ai_core.stop()

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API ENDPOINTS ---

@app.post("/api/ai/toggle")
async def toggle_ai(active: bool):
    """
    CRITICAL: This controls the Orchestrator which the loop listens to.
    """
    if active:
        logger.info("🔌 API COMMAND: ACTIVATE SYSTEM")
        ai_core.start() # Sets orchestrator.is_running = True
        return {"status": "success", "message": "SYSTEM ONLINE", "active": True}
    else:
        logger.info("🔌 API COMMAND: SHUTDOWN SYSTEM")
        ai_core.stop() # Sets orchestrator.is_running = False
        return {"status": "success", "message": "SYSTEM OFFLINE", "active": False}

@app.get("/api/ai/state")
async def get_ai_state():
    # Returns the TRUE state of the orchestrator
    is_running = False
    if hasattr(ai_core, 'orchestrator'):
        is_running = ai_core.orchestrator.is_running
        
    return {
        "running": is_running,
        "mode": ai_core.state["mode"],
        "engine": "GEN-4 (Connected)"
    }

@app.get("/api/system/logs")
async def get_logs():
    # Return the actual captured logs from the buffer
    return LOG_BUFFER
