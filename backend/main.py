# backend/main.py
"""
REDLINE BACKEND — GEN 2.5 (NON-CUSTODIAL)
FastAPI entrypoint
Integruje: Live Feed, Indicators, IAM, API Key Management, n8n Support
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
from backend.ai_config_manager import AIConfigManager
from core.orchestrator import Orchestrator
from core.logger import get_latest_logs

try:
    from data.feed import DataFeed
    FEED_AVAILABLE = True
except ImportError:
    print(" WARNING: DataFeed not found. Scanner will use mock data.")
    FEED_AVAILABLE = False

app = FastAPI(title="REDLINE API", version="2.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"], # Allow * for n8n locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL AI CONFIGURATION (Managed Persistence)
AI_CONFIG = AIConfigManager.load_config()

ai_core = RedlineAICore(mode="PAPER", timeframe="1h")
orchestrator = Orchestrator(mode="PAPER", ai_core=ai_core)

# =====================================================
# BACKGROUND TASK
# =====================================================
async def trading_background_loop():
    """
    Infinite loop running in background.
    """
    print(" 🚀 STARTING BACKGROUND TRADING LOOP...")
    orchestrator.is_running = True
    while orchestrator.is_running:
        try:
            # We pass current config to the cycle if needed, but orchestrator currently handles its own logic.
            # In V3 this might call run(AI_CONFIG)
            orchestrator.run_cycle()
        except Exception as e:
            print(f"  ORCHESTRATOR ERROR: {e}")
        await asyncio.sleep(60) # Cycle every 1 minute for now

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(trading_background_loop())
    asyncio.create_task(binance_ws_loop())
    asyncio.create_task(whale_poller_loop())

async def whale_poller_loop():
    """
    Background Task: Scan tracked whales for new on-chain activity.
    """
    print(" 🐋 WHALE POLLER STARTED")
    while True:
        try:
            # Run blocking I/O in thread pool to not block asyncio
            await asyncio.to_thread(ai_core.whale_watcher.scan_chain_activity)
        except Exception as e:
            print(f" 🐋 Poll Error: {e}")
            
        await asyncio.sleep(60) # Chech every 60s

async def binance_ws_loop():
    """
    Subscribes to !miniTicker@arr via WebSocket to feed the Scalper Engine.
    """
    """
    Subscribes to !miniTicker@arr via WebSocket to feed the Scalper Engine.
    """
    import json
    import websockets
    
    uri = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
    print(" 🔌 CONNECTING TO BINANCE WEBSOCKET...")
    
    while True:
        try:
            async with websockets.connect(uri) as ws:
                print(" ✅ WS CONNECTED (Real-Time Data)")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    # Filtrujemy tylko nasze monety, żeby nie zalać Orchestratora
                    for ticker in data:
                        sym = ticker['s']
                        if sym.endswith("USDT") and orchestrator.is_running:
                             # Wywołujemy event w Orchestratorze (Level 5)
                            tick_payload = {
                                'symbol': sym, # e.g. BTCUSDT (Binance format) -> trzeba skonwertować na BTC/USDT?
                                'price': float(ticker['c']),
                                'vol': float(ticker['v'])
                            }
                            # Konwersja formatu symbolu
                            std_sym = sym.replace("USDT", "/USDT")
                            tick_payload['symbol'] = std_sym
                            
                            orchestrator.on_tick(tick_payload)
                            
        except Exception as e:
            print(f" ⚠️ WS DISCONNECTED: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

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
    risk_mode: str = "BALANCED"
    sentiment_weight: float = 50.0
    max_open_positions: int = 3
    auto_trade_enabled: bool = False
    confirmation_required: bool = True
    news_impact_enabled: bool = True
    explainability: bool = True # Legacy support

class N8nWebhookRequest(BaseModel):
    source: str = "n8n"
    type: str = "sentiment" # sentiment, news, alert
    value: float = 50.0
    summary: str = ""

class OrderRequest(BaseModel):
    symbol: str
    side: str
    amount: float

# =====================================================
# ENDPOINTS
# =====================================================

# --- WEBSOCKETS ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Inject manager into orchestrator so it can emit events
orchestrator.ws_manager = manager


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
    global AI_CONFIG
    AI_CONFIG = AIConfigManager.load_config() # Reload from disk
    return AI_CONFIG

@app.post("/api/admin/ai_settings")
def update_ai_settings(req: AISettingsRequest):
    global AI_CONFIG
    updates = req.dict()
    AI_CONFIG = AIConfigManager.update_config(updates)
    return {"status": "UPDATED", "config": AI_CONFIG}

# --- WEBHOOKS: n8n / EXTERNAL ---
@app.post("/api/webhooks/external_data")
def external_data_webhook(req: N8nWebhookRequest):
    """
    Endpoint for n8n to push sentiment or news analysis.
    """
    print(f"📡 [WEBHOOK] Received from {req.source}: {req.summary} (Val: {req.value})")
    
    # Update AI Core Context
    ai_core.update_external_context({
        "sentiment": req.value,
        "summary": req.summary
    })
    
    return {"status": "ACCEPTED", "context_updated": True}


# --- USER SETTINGS ---
class ConnectExchangeRequest(BaseModel):
    username: str
    exchange: str
    api_key: str
    api_secret: str

@app.post("/api/admin/reset_system")
def reset_system():
    """
    Hard reset for testing: Balance -> 1000, History -> Clear.
    """
    conn = UserManager.load_db_connection()
    try:
        conn.execute("UPDATE wallet SET balance = 1000.0 WHERE id = 1")
        conn.execute("DELETE FROM trade_history")
        conn.execute("DELETE FROM assets")
        conn.commit()
        return {"status": "RESET_COMPLETE", "balance": 1000.0}
    except Exception as e:
        raise HTTPException(500, f"Reset Failed: {e}")
    finally:
        conn.close()

@app.post("/api/user/connect_exchange")
def connect_exchange(req: ConnectExchangeRequest):
    """
    Allow user to connect their exchange API keys.
    """
    # In a real app, verifying the username matches the token is crucial.
    # For this prototype, we trust the username matches the session.
    updates = {
        "exchange_config": {
            "exchange": req.exchange,
            "api_key": req.api_key,
            "api_secret": req.api_secret # Should be encrypted
        }
    }
    # We use the UserManager to save this to the JSON db
    db = UserManager.load_db()
    if req.username in db.get('active', {}):
        # Merge update
        user_data = db['active'][req.username]
        user_data['exchange_config'] = updates['exchange_config']
        UserManager.save_db(db)
        return {"status": "CONNECTED", "msg": f"Successfully connected to {req.exchange}"}
    
    raise HTTPException(404, "User not found")

# --- SCANNER ---
@app.post("/api/scanner/run_cycle")
async def run_cycle_manual():
    """Manually triggers a full orchestrator cycle."""
    try:
        # Run the full god-mode mission demo
        await orchestrator.run_demonstration_mission()
        return {"status": "CYCLE_COMPLETED", "timestamp": datetime.now()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Orchestrator Error: {e}")

@app.get("/api/scanner/run")
def run_scanner():
    """
    Runs the REAL Orchestrator Scanner (Parallel Multi-Asset).
    Uses the persistent AI CONFIG.
    """
    if not FEED_AVAILABLE:
        return [{"symbol": "MOCK/BTC", "signal": "FEED ERROR", "price": 0, "rsi": 0, "confidence": 0, "reason": "No data feed"}]

    # Refresh config before running
    current_config = AIConfigManager.load_config()
    
    # Run the real scanner logic
    # This uses ai_core.evaluate() which now respects external_context and config
    try:
        results = orchestrator.scanner.run(current_config)
        return results if results else []
    except Exception as e:
        print(f"Scanner Logic Error: {e}")
        # Fallback to empty if scan crashes
        return []

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
def ai_state():
    state = ai_core.get_state()
    state["orchestrator_running"] = orchestrator.is_running
    return state

@app.post("/api/ai/toggle")
async def toggle_ai(active: bool):
    """
    Switches the AI Core AUTOPILOT on or off.
    """
    print(f"[API] Toggling AI Core: {active}")
    orchestrator.set_autopilot(active)
    # If activating, ensure the loop is running (it might have exited if it was previously off)
    if active:
        # Trigger an immediate cycle so user sees action right away
        asyncio.create_task(orchestrator.run_demonstration_mission())
    return {"status": "UPDATED", "active": orchestrator.is_running}

@app.get("/api/system/logs")
def system_logs():
    """Returns the latest 50 system logs for the HUD."""
    return get_latest_logs(50)

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
                "ai_running": ai_core.state["running"],
                "ext_context": ai_core.external_context.get("summary", "")[:20] + "..." # Small info on HUD
            }

            await ws.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect: pass

# =====================================================
# WHALE WATCHER API
# =====================================================

class WhaleAddRequest(BaseModel):
    address: str
    label: str
    network: str = "ETH"

class WhaleSignalRequest(BaseModel):
    symbol: str
    side: str
    whale_address: str

@app.get("/api/whales")
def get_whales():
    """List all watched whales."""
    return ai_core.whale_watcher.get_whales()

@app.post("/api/whales")
def add_whale(req: WhaleAddRequest):
    """Add a new whale to watch."""
    success, msg = ai_core.whale_watcher.add_whale(req.address, req.label, req.network)
    if not success:
        raise HTTPException(400, msg)
    return {"status": "ADDED", "msg": msg}

@app.delete("/api/whales/{address}")
def remove_whale(address: str):
    """Remove a whale."""
    if ai_core.whale_watcher.remove_whale(address):
        return {"status": "REMOVED"}
    raise HTTPException(404, "Whale not found")

@app.post("/api/whales/inject_signal")
def inject_whale_signal(req: WhaleSignalRequest):
    """
    Simulation Endpoint: Force specific whale to signal a move.
    """
    ai_core.whale_watcher.inject_signal(req.symbol, req.side, req.whale_address)
    return {"status": "INJECTED", "msg": f"Whale {req.whale_address} signalled {req.side} on {req.symbol}"}
