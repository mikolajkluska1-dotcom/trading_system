# backend/main.py
"""
REDLINE BACKEND — GEN 2.2 (AUTH UPDATE)
FastAPI entrypoint

"""


import sys
import os
import asyncio
import psutil
import random
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# PATH FIX
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
# APP SETUP
# =====================================================
app = FastAPI(
    title="REDLINE API",
    version="2.2",
    description="AI Trading & Risk Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CORE INSTANCES
# =====================================================
ai_core = RedlineAICore(mode="PAPER", timeframe="1h")

# =====================================================
# DTO (Data Transfer Objects)
# =====================================================
class LoginRequest(BaseModel):
    username: str
    password: str

# NOWY MODEL DLA ROZBUDOWANEJ REJESTRACJI
class RegisterRequest(BaseModel):
    fullName: str
    phone: str
    email: str
    about: str

class OrderRequest(BaseModel):
    symbol: str
    side: str
    amount: float

class UserActionRequest(BaseModel):
    username: str
    role: str = "USER"

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
        "token": "mock-jwt-token-access-granted",
        "login_time": datetime.utcnow().isoformat()
    }

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    """
    Obsługa wniosku o dostęp z rozbudowanymi danymi.
    Mapujemy dane formularza na strukturę bazy UserManagera.
    """
    # Username to Email (dla unikalności)
    username = req.email
    # Hasło tymczasowe (użytkownik i tak czeka na akceptację admina)
    temp_password = "PENDING_APPROVAL"
    
    # Sklejamy dane kontaktowe w jeden czytelny string dla Admina
    contact_info = f"Name: {req.fullName} | Phone: {req.phone} | Note: {req.about}"

    success, msg = UserManager.request_account(username, temp_password, contact_info)
    
    if not success:
        # Jeśli taki email już jest w bazie
        raise HTTPException(status_code=400, detail=msg)
        
    return {"status": "REQUEST_SUBMITTED", "message": "Application received"}

# =====================================================
# ADMIN API
# =====================================================
@app.get("/api/admin/users")
def get_users():
    db = UserManager.load_db()
    clean_active = {u: {k: v for k, v in data.items() if k != 'hash'} for u, data in db.get('active', {}).items()}
    clean_pending = {u: {k: v for k, v in data.items() if k != 'hash'} for u, data in db.get('pending', {}).items()}
    return {"active": clean_active, "pending": clean_pending}

@app.post("/api/admin/approve")
def approve_user(req: UserActionRequest):
    if UserManager.approve_user(req.username, req.role):
        return {"status": "APPROVED", "user": req.username, "role": req.role}
    raise HTTPException(status_code=400, detail="User not found in pending")

@app.post("/api/admin/reject")
def reject_user(req: UserActionRequest):
    if UserManager.reject_user(req.username):
        return {"status": "REJECTED", "user": req.username}
    raise HTTPException(status_code=400, detail="User not found in pending")

# =====================================================
# WALLET & TRADING
# =====================================================
@app.get("/api/wallet")
def wallet():
    return WalletManager.get_wallet_data()

@app.get("/api/trading/assets")
def wallet_assets():
    data = WalletManager.get_wallet_data()
    return data.get("assets", [])

@app.get("/api/trading/positions")
def get_positions():
    return [
        {"symbol": "BTC/USDT", "side": "LONG", "size": 0.5, "pnl": 120.50},
        {"symbol": "ETH/USDT", "side": "SHORT", "size": 10.0, "pnl": -45.00}
    ]

@app.post("/api/trading/order")
def execute_order(order: OrderRequest):
    return {
        "status": "FILLED",
        "order_id": f"ORD-{random.randint(1000,9999)}",
        "price": random.randint(40000, 60000),
        "symbol": order.symbol
    }

# =====================================================
# SCANNER & AI
# =====================================================
@app.get("/api/scanner/run")
def run_scanner():
    return [
        {"symbol": "SOL/USDT", "signal": "STRONG BUY", "confidence": 0.89},
        {"symbol": "XRP/USDT", "signal": "WEAK SELL", "confidence": 0.65}
    ]

# =====================================================
# ML / NEURAL LABS
# =====================================================
@app.get("/api/ml/status")
def ml_status():
    return {
        "model_version": "2.0.4-alpha",
        "current_epoch": 45,
        "total_epochs": 100,
        "accuracy": 0.78,
        "loss": 0.3421
    }

@app.get("/api/ml/chart")
def ml_chart():
    data = []
    for i in range(1, 50):
        data.append({
            "epoch": i,
            "loss": max(0.1, 1.0 - (i * 0.02) + random.uniform(-0.05, 0.05)),
            "accuracy": min(0.95, 0.5 + (i * 0.01) + random.uniform(-0.02, 0.02))
        })
    return data

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

# =====================================================
# HUD
# =====================================================
@app.websocket("/ws/hud")
async def hud_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            wallet_data = WalletManager.get_wallet_data()
            payload = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": psutil.cpu_percent(),
                "mem": psutil.virtual_memory().percent,
                "funds": wallet_data.get("balance", 0.0),
                "ai_mode": ai_core.state["mode"],
                "ai_running": ai_core.state["running"]
            }
          
            await ws.send_json(payload)
            await asyncio.sleep(1)
    
    except WebSocketDisconnect:
       pass