# backend/main.py
"""
REDLINE BACKEND — GEN 2
FastAPI entrypoint
AI-first architecture (no UI logic here)
"""

import sys
import os
import asyncio
import psutil
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# PATH FIX (żeby widzieć core/ml/trading)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# =====================================================
# IMPORTY SYSTEMU
# =====================================================
from security.user_manager import UserManager
from trading.wallet import WalletManager
from backend.ai_core import RedlineAICore

# =====================================================
# APP
# =====================================================
app = FastAPI(
    title="REDLINE API",
    version="2.0",
    description="AI Trading & Risk Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# AI CORE (SINGLE INSTANCE)
# =====================================================
ai_core = RedlineAICore(mode="PAPER", timeframe="1h")

# =====================================================
# MODELE
# =====================================================
class LoginRequest(BaseModel):
    username: str
    password: str

# =====================================================
# SYSTEM / HEALTH
# =====================================================
@app.get("/api/status")
def system_status():
    return {
        "status": "ONLINE",
        "engine": "REDLINE_AI_CORE_GEN2",
        "time": datetime.utcnow().isoformat()
    }

# =====================================================
# AUTH
# =====================================================
@app.post("/api/auth/login")
def login(req: LoginRequest):
    role = UserManager.verify_login(req.username, req.password)
    if not role:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "user": req.username,
        "role": role,
        "login_time": datetime.utcnow().isoformat()
    }

# =====================================================
# WALLET
# =====================================================
@app.get("/api/wallet")
def wallet():
    return WalletManager.get_wallet_data()

# =====================================================
# AI CORE CONTROL
# =====================================================
@app.get("/api/ai/state")
def ai_state():
    return ai_core.get_state()

@app.post("/api/ai/mode/{mode}")
def set_ai_mode(mode: str):
    try:
        ai_core.set_mode(mode.upper())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ai_core.get_state()

@app.post("/api/ai/run")
def run_ai_once():
    """
    Manual trigger of single AI cycle
    """
    return ai_core.run_once()

@app.post("/api/ai/auto/start")
def start_auto():
    ai_core.start_auto(interval_sec=3600)
    return {
        "status": "AUTO_MODE_STARTED",
        "interval_sec": 3600
    }

@app.post("/api/ai/auto/stop")
def stop_auto():
    ai_core.stop_auto()
    return {
        "status": "AUTO_MODE_STOPPED"
    }

# =====================================================
# HUD — REALTIME SYSTEM METRICS
# =====================================================
@app.websocket("/ws/hud")
async def hud_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            wallet = WalletManager.get_wallet_data()

            payload = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": psutil.cpu_percent(),
                "mem": psutil.virtual_memory().percent,
                "funds": wallet.get("balance", 0.0),
                "ai_mode": ai_core.state["mode"],
                "ai_running": ai_core.state["running"]
            }

            await ws.send_json(payload)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
