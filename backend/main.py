# backend/main.py
"""
REDLINE BACKEND — GEN 2.5 (NON-CUSTODIAL)
FastAPI entrypoint
Integruje: Live Feed, Indicators, IAM, API Key Management
"""


import sys
import os
import asyncio
import psutil
import random
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from security.user_manager import UserManager
from trading.wallet import WalletManager
from backend.ai_core import RedlineAICore

try:
    from data.feed import DataFeed
    FEED_AVAILABLE = True
except ImportError:
    print(" WARNING: DataFeed not found. Scanner will use mock data.")
    FEED_AVAILABLE = False

app = FastAPI(title="REDLINE API", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL AI CONFIGURATION
AI_CONFIG = {
    "min_confidence": 0.60,
    "volatility_filter": False,
    "portfolio_mode": False,
    "explainability": True
}

ai_core = RedlineAICore(mode="PAPER", timeframe="1h")

# =====================================================
# DTO MODELS
# =====================================================
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    fullName: str
    phone: str
    email: str
    about: str

class UserActionRequest(BaseModel):
    username: str
    role: str = "USER"

# Zaktualizowany model: teraz przyjmuje klucze API
class UserUpdateRequest(BaseModel):
    username: str
    trading_enabled: bool
    risk_limit: float
    notes: str
    api_key: str = None     # Opcjonalne
    api_secret: str = None  # Opcjonalne

class AISettingsRequest(BaseModel):
    min_confidence: float
    volatility_filter: bool
    portfolio_mode: bool
    explainability: bool

class OrderRequest(BaseModel):
    symbol: str
    side: str
    amount: float



# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/api/status")
def system_status():
    return {
        "status": "ONLINE",
        "engine": "REDLINE_AI_CORE_GEN2.5",
        "feed": "LIVE" if FEED_AVAILABLE else "MOCK",
        "ai_config": AI_CONFIG
    }

# --- AUTH ---
@app.post("/api/auth/login")
def login(req: LoginRequest):
    role = UserManager.verify_login(req.username, req.password)
    if not role: raise HTTPException(401, "Invalid credentials")
    return {"user": req.username, "role": role, "token": "mock-jwt"}

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    contact = f"{req.fullName} | {req.phone} | {req.about}"
    success, msg = UserManager.request_account(req.email, "PENDING_APPROVAL", contact)
    if not success: raise HTTPException(400, msg)
    return {"status": "SUBMITTED"}

# --- ADMIN: USERS ---
@app.get("/api/admin/users")
def get_users():
    db = UserManager.load_db()
    # Maskujemy klucze API przed wysłaniem do frontendu dla bezpieczeństwa
    active_users = {}
    for u, data in db.get('active', {}).items():
        clean_data = {k: v for k, v in data.items() if k != 'hash'}
        ex_config = clean_data.get('exchange_config', {})
        
        # Flaga dla frontendu, czy klucze są ustawione
        clean_data['has_api_key'] = bool(ex_config.get('api_key'))
        clean_data['exchange_name'] = ex_config.get('exchange', 'NONE')
        
        # Usuwamy surowe klucze z odpowiedzi
        if 'exchange_config' in clean_data:
            del clean_data['exchange_config']
        active_users[u] = clean_data

    clean_pending = {u: {k: v for k, v in data.items() if k != 'hash'} for u, data in db.get('pending', {}).items()}
    return {"active": active_users, "pending": clean_pending}

@app.post("/api/admin/approve")
def approve_user(req: UserActionRequest):
    if UserManager.approve_user(req.username, req.role): return {"status": "APPROVED"}
    raise HTTPException(400, "User not found")

@app.post("/api/admin/reject")
def reject_user(req: UserActionRequest):
    if UserManager.reject_user(req.username): return {"status": "REJECTED"}
    raise HTTPException(400, "User not found")

@app.post("/api/admin/update_user")
def update_user(req: UserUpdateRequest):
    """
    Admin ustawia ryzyko oraz KLUCZE API użytkownika.
    """
    updates = {
        "trading_enabled": req.trading_enabled,
        "risk_limit": req.risk_limit,
        "notes": req.notes
    }
    # Dodajemy klucze tylko jeśli zostały wpisane (nie są puste)
    if req.api_key and req.api_secret:
        updates["api_key"] = req.api_key
        updates["api_secret"] = req.api_secret

    if UserManager.update_user_settings(req.username, updates): 
        return {"status": "UPDATED"}
    
    raise HTTPException(400, "Update failed")

# --- ADMIN: AI GOVERNANCE ---
@app.get("/api/admin/ai_settings")
def get_ai_settings():
    return AI_CONFIG

@app.post("/api/admin/ai_settings")
def update_ai_settings(req: AISettingsRequest):
    global AI_CONFIG
    AI_CONFIG["min_confidence"] = req.min_confidence
    AI_CONFIG["volatility_filter"] = req.volatility_filter
    AI_CONFIG["portfolio_mode"] = req.portfolio_mode
    AI_CONFIG["explainability"] = req.explainability
    return {"status": "UPDATED", "config": AI_CONFIG}

# --- SCANNER ---
@app.get("/api/scanner/run")
def run_scanner():
    if not FEED_AVAILABLE:
        return [{"symbol": "MOCK/BTC", "signal": "FEED ERROR", "price": 0, "rsi": 0, "confidence": 0, "reason": "No data feed"}]

    watchlist = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    results = []

    for symbol in watchlist:
        try:
            df = DataFeed.get_market_data(symbol, tf="1h", limit=50)
            if df.empty: continue

            last = df.iloc[-1]
            price = float(last['close'])
            rsi = float(last['rsi']) if 'rsi' in df.columns else 50.0
            
            signal = "NEUTRAL"
            reason = "Market is ranging."
            confidence = 0.50

            if rsi < 30:
                signal = "STRONG BUY"
                reason = f"Oversold (RSI {rsi:.1f})."
                confidence = 0.85
            elif rsi > 70:
                signal = "STRONG SELL"
                reason = f"Overbought (RSI {rsi:.1f})."
                confidence = 0.85

            if AI_CONFIG["volatility_filter"]:
                volatility = (last['high'] - last['low']) / last['close']
                if volatility > 0.02:
                    signal = "NEUTRAL"
                    reason = "VOLATILITY FILTER: High Risk."
                    confidence = 0.0

            if confidence < AI_CONFIG["min_confidence"]:
                signal = "NEUTRAL"
                reason += " [Low Confidence]"

            results.append({
                "symbol": symbol,
                "price": price,
                "signal": signal,
                "rsi": round(rsi, 2),
                "confidence": confidence,
                "reason": reason if AI_CONFIG["explainability"] else "AI Decision"
            })

        except Exception as e:
            continue

    return results

# --- DATA ENDPOINTS ---
@app.get("/api/wallet")
def wallet(): return WalletManager.get_wallet_data()

@app.get("/api/trading/assets")
def wallet_assets(): return WalletManager.get_wallet_data().get("assets", [])

@app.get("/api/trading/positions")
def get_positions(): return []

@app.post("/api/trading/order")
def execute_order(order: dict): return {"status": "FILLED", "order_id": "MOCK-123"}

@app.get("/api/ml/status")
def ml_status(): return {"model_version": "2.5.0", "accuracy": 0.82, "loss": 0.28}

@app.get("/api/ml/chart")
def ml_chart(): return [{"epoch": i, "loss": 1.0/(i+1), "accuracy": 0.5 + (i/200)} for i in range(1, 50)]

@app.get("/api/ai/state")
def ai_state(): return ai_core.get_state()

@app.websocket("/ws/hud")
async def hud_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # W przyszłości tu będziemy pytać Binance o prawdziwe saldo API Key
            wallet_data = WalletManager.get_wallet_data() 
            payload = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "cpu": psutil.cpu_percent(),
                "mem": psutil.virtual_memory().percent,
                "funds": wallet_data.get("balance", 0.0), # To będzie balance z Binance
                "ai_mode": "GEN-3 ACTIVE",
                "ai_running": ai_core.state["running"]
            }
         
            await ws.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect: pass