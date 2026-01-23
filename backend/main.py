import sys
import os
import subprocess

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

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import requests
import json
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

# --- TRADER PROCESS MANAGEMENT ---
# Global variable to track the background trading script
trader_process = None

# --- API ENDPOINTS ---

@app.post("/api/trader/start")
async def start_trader():
    """
    Starts the serious_trader.py script in the background.
    """
    global trader_process
    
    # Check if already running
    if trader_process is not None and trader_process.poll() is None:
        return {
            "status": "active",
            "message": "System AI już działa.",
            "running": True
        }
    
    try:
        # Start the trader script as a background process
        # Using auto_trader.py (you can rename it to serious_trader.py if needed)
        trader_process = subprocess.Popen(
            [sys.executable, "backend/auto_trader.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"🚀 TRADER PROCESS STARTED (PID: {trader_process.pid})")
        return {
            "status": "active",
            "message": "System AI uruchomiony.",
            "running": True,
            "pid": trader_process.pid
        }
    except Exception as e:
        logger.error(f"❌ Failed to start trader: {e}")
        raise HTTPException(status_code=500, detail=f"Nie udało się uruchomić tradera: {str(e)}")

@app.post("/api/trader/stop")
async def stop_trader():
    """
    Stops the running trader script.
    """
    global trader_process
    
    if trader_process is None or trader_process.poll() is not None:
        return {
            "status": "inactive",
            "message": "System AI nie był uruchomiony.",
            "running": False
        }
    
    try:
        trader_process.terminate()
        trader_process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
        logger.info("🛑 TRADER PROCESS TERMINATED")
        trader_process = None
        return {
            "status": "inactive",
            "message": "System AI zatrzymany.",
            "running": False
        }
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't terminate gracefully
        trader_process.kill()
        trader_process = None
        logger.warning("⚠️ TRADER PROCESS FORCE KILLED")
        return {
            "status": "inactive",
            "message": "System AI zatrzymany (wymuszony).",
            "running": False
        }
    except Exception as e:
        logger.error(f"❌ Failed to stop trader: {e}")
        raise HTTPException(status_code=500, detail=f"Błąd zatrzymywania tradera: {str(e)}")

@app.get("/api/trader/status")
async def get_trader_status():
    """
    Returns the current status of the trader process.
    """
    global trader_process
    
    is_running = trader_process is not None and trader_process.poll() is None
    
    return {
        "running": is_running,
        "status": "active" if is_running else "inactive",
        "pid": trader_process.pid if is_running else None
    }

# --- EXISTING ENDPOINTS ---

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
    # Returns the TRUE state of the orchestrator + Memory
    return ai_core.get_state()

@app.get("/api/system/logs")
async def get_logs():
    # Return the actual captured logs from the buffer
    return LOG_BUFFER

# --- DATA MODELS ---
class WebhookPayload(BaseModel):
    source: str       # e.g., "n8n", "telegram"
    type: str = "generic" # e.g., "sentiment", "command"
    value: Optional[Union[str, float]] = None
    summary: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

@app.post("/api/webhook")
async def external_webhook(data: WebhookPayload):
    """
    Universal entry point for n8n automation.
    """
    logger.info(f"📡 WEBHOOK RECEIVED [{data.source}]: {data.type}")

    # 1. TELEGRAM COMMANDS (Immediate System Control)
    if data.source == "telegram" and data.type == "command" and data.payload:
        cmd = data.payload.get("command", "").upper()
        if cmd == "/STOP" or cmd == "/PANIC":
            ai_core.stop()
            return {"status": "executed", "message": "SYSTEM HALTED BY REMOTE COMMAND"}
        elif cmd == "/START":
            ai_core.start()
            return {"status": "executed", "message": "SYSTEM RESUMED"}
    
    # 2. DATA INGESTION (n8n, Sentiment, Whales)
    # Pass everything else to the AI Core Brain
    try:
        # Convert Pydantic model to dict for flexibility
        payload_dict = data.dict()
        ai_core.process_webhook_data(payload_dict)
        return {"status": "processed", "message": "Data fed to AI Core"}
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return {"status": "error", "message": str(e)}

# --- CHATBOT LOGIC ---
class ChatRequest(BaseModel):
    message: str

def get_db_connection():
    """Intelligent database connection - works in Docker AND locally"""
    import psycopg2
    import time
    
    # Próba 1: Połączenie wewnątrz Dockera (używamy nazwy serwisu "timescaledb" i portu wewnętrznego 5432)
    try:
        conn = psycopg2.connect(
            host="timescaledb",  # To jest nazwa kontenera w sieci Docker
            port="5432",         # Port WEWNĘTRZNY (nie 5435!)
            database="redline_db",
            user="redline_user",
            password="redline_pass"
        )
        return conn
    except Exception as e_docker:
        # Próba 2: Jeśli nie jesteśmy w Dockerze, tylko testujemy lokalnie (używamy localhost i portu 5435)
        try:
            conn = psycopg2.connect(
                host="localhost",
                port="5435",     # Twój port ZEWNĘTRZNY
                database="redline_db",
                user="redline_user",
                password="redline_pass"
            )
            return conn
        except Exception as e_local:
            print(f"❌ BŁĄD BAZY DANYCH: Nie udało się połączyć ani przez Docker, ani localhost.")
            print(f"   Docker error: {e_docker}")
            print(f"   Local error: {e_local}")
            raise e_local

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    user_msg = request.message.lower()
    
    # 1. KONFIGURACJA POŁĄCZENIA Z LM STUDIO (Host Windows widziany z Dockera)
    LLM_URL = "http://host.docker.internal:1234/v1/chat/completions"
    HEADERS = {"Content-Type": "application/json"}
    
    # Domyślna odpowiedź awaryjna
    bot_response = "⚠️ System AI offline. Nie mogę połączyć się z modelem."
    
    # 2. SMART MAPPING (Polskie nazwy -> Symbol Binance)
    COIN_MAPPING = {
        "btc": "BTC", "bitcoin": "BTC", "bitcoina": "BTC", "bitkojna": "BTC",
        "eth": "ETH", "ethereum": "ETH", "eter": "ETH",
        "sol": "SOL", "solana": "SOL", "solany": "SOL",
        "bnb": "BNB", "binance": "BNB",
        "xrp": "XRP", "ripple": "XRP",
        "ada": "ADA", "cardano": "ADA",
        "doge": "DOGE", "dogecoin": "DOGE"
    }
    
    detected_symbol = None
    for word in user_msg.replace("?", "").replace(".", "").split():
        if word in COIN_MAPPING:
            detected_symbol = COIN_MAPPING[word]
            break
    
    # 3. LOGIKA BAZY DANYCH (Zabezpieczenie przed halucynacją)
    price_context = ""
    is_price_query = False
    
    if detected_symbol:
        try:
            symbol_pair = f"{detected_symbol}/USDT"
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT close FROM market_candles WHERE symbol = '{symbol_pair}' ORDER BY time DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                price = result[0]
                # BARDZO WAŻNE: Wymuszamy format odpowiedzi w System Prompt
                price_context = f"AKTUALNA CENA Z BAZY DANYCH: {detected_symbol} = {price:.2f} USDT."
                is_price_query = True
            else:
                price_context = "BRAK DANYCH W BAZIE. Nie zmyślaj ceny."
        except Exception as e:
            price_context = f"BŁĄD BAZY: {str(e)}. Nie zmyślaj ceny."
    
    # 4. TURBO PROMPT (Dla szybkości)
    if is_price_query:
        system_prompt = (
            f"Jesteś Redline. {price_context} "
            "Podaj tę cenę użytkownikowi i dodaj krótki, cwaniacki komentarz po polsku. "
            "Maksymalnie 2 zdania."
        )
    else:
        system_prompt = (
            "Jesteś Redline. Mówisz krótko, zwięźle i po polsku. "
            "Nie tłumacz się. Odpowiadaj jak ekspert."
        )

    # 5. WYSŁANIE ZAPYTANIA DO AI
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ],
        "temperature": 0.3,   # Mała kreatywność = szybsze myślenie
        "max_tokens": 60      # LIMIT SŁÓW! Ucinamy go szybko, żeby nie mulił.
    }

    try:
        # Timeout 20s żeby nie blokować interfejsu
        response = requests.post(LLM_URL, headers=HEADERS, json=payload, timeout=200)
        
        if response.status_code == 200:
            data = response.json()
            bot_response = data['choices'][0]['message']['content']
        else:
            bot_response = f"❌ Błąd LLM ({response.status_code}): {response.text}"
            
    except requests.exceptions.ConnectionError:
        # FALLBACK: Jeśli LM Studio nie działa, zwróć chociaż cenę z bazy
        if is_price_query:
            bot_response = f"{price_context} (⚠️ Moduł AI Offline - uruchom LM Studio 'Server')."
        else:
            bot_response = "🔌 Błąd połączenia z mózgiem AI. Upewnij się, że serwer LM Studio jest aktywny."
    except Exception as e:
        bot_response = f"❌ Krytyczny błąd: {str(e)}"

    return {"response": bot_response}
