import sys
import os
import subprocess

# --- CRITICAL WINDOWS FIX ---
# Forces UTF-8 encoding to prevent crashes when printing emojis (ð§ , ð°)
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
from pydantic import BaseModel
from agents.BackendAPI.security.user_manager import UserManager
import asyncio
import logging
import requests
import json
from contextlib import asynccontextmanager

# IMPORTS
from agents.BackendAPI.backend.ai_core import RedlineAICore
from agents.BackendAPI.backend.events_server import start_background_loop
from agents.Database.core.db import Database

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
    logger.info("ð SYSTEM STARTUP INITIATED")
    Database.initialize()
    
    # Start the background loop thread
    loop_task = asyncio.create_task(start_background_loop(ai_core))
    
    yield
    
    # SHUTDOWN
    logger.info("ð SHUTDOWN SIGNAL RECEIVED")
    ai_core.stop()

app = FastAPI(lifespan=lifespan)

# --- AUTH MODELS ---
class LoginRequest(BaseModel):
    username: str
    password: str

# --- AUTH ENDPOINTS ---
@app.post("/api/auth/login")
async def login(req: LoginRequest):
    try:
        role = UserManager.verify_login(req.username, req.password)
        if role:
            return {
                "success": True, 
                "username": req.username, 
                "role": role,
                "message": "Login successful"
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/me")
async def get_current_user(username: str):
    try:
        db = UserManager.load_db()
        user = db['active'].get(username)
        if user:
            return {
                "success": True,
                "username": username,
                "role": user.get('role'),
                "contact": user.get('contact'),
                "trading_enabled": user.get('trading_enabled')
            }
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        print(f"Me error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            "message": "System AI juÅ¼ dziaÅa.",
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
        logger.info(f"ð TRADER PROCESS STARTED (PID: {trader_process.pid})")
        return {
            "status": "active",
            "message": "System AI uruchomiony.",
            "running": True,
            "pid": trader_process.pid
        }
    except Exception as e:
        logger.error(f"â Failed to start trader: {e}")
        raise HTTPException(status_code=500, detail=f"Nie udaÅo siÄ uruchomiÄ tradera: {str(e)}")

@app.post("/api/trader/stop")
async def stop_trader():
    """
    Stops the running trader script.
    """
    global trader_process
    
    if trader_process is None or trader_process.poll() is not None:
        return {
            "status": "inactive",
            "message": "System AI nie byÅ uruchomiony.",
            "running": False
        }
    
    try:
        trader_process.terminate()
        trader_process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
        logger.info("ð TRADER PROCESS TERMINATED")
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
        logger.warning("â ï¸ TRADER PROCESS FORCE KILLED")
        return {
            "status": "inactive",
            "message": "System AI zatrzymany (wymuszony).",
            "running": False
        }
    except Exception as e:
        logger.error(f"â Failed to stop trader: {e}")
        raise HTTPException(status_code=500, detail=f"BÅÄd zatrzymywania tradera: {str(e)}")

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
        logger.info("ð API COMMAND: ACTIVATE SYSTEM")
        ai_core.start() # Sets orchestrator.is_running = True
        return {"status": "success", "message": "SYSTEM ONLINE", "active": True}
    else:
        logger.info("ð API COMMAND: SHUTDOWN SYSTEM")
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

# --- AUTHENTICATION ---
from agents.BackendAPI.security.user_manager import UserManager

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    contact: str

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """
    Authenticate user with username and password.
    Returns user data with role if successful.
    """
    try:
        role = UserManager.verify_login(req.username, req.password)
        if not role:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        from datetime import datetime
        return {
            "user": req.username,
            "role": role,
            "login_time": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """
    Request a new user account (requires admin approval).
    """
    try:
        success, message = UserManager.request_account(req.username, req.password, req.contact)
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return {"status": "success", "message": message}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
    logger.info(f"ð¡ WEBHOOK RECEIVED [{data.source}]: {data.type}")

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
    
    # PrÃ³ba 1: PoÅÄczenie wewnÄtrz Dockera (uÅ¼ywamy nazwy serwisu "timescaledb" i portu wewnÄtrznego 5432)
    try:
        conn = psycopg2.connect(
            host="timescaledb",  # To jest nazwa kontenera w sieci Docker
            port="5432",         # Port WEWNÄTRZNY (nie 5435!)
            database="redline_db",
            user="redline_user",
            password="redline_pass"
        )
        return conn
    except Exception as e_docker:
        # PrÃ³ba 2: JeÅli nie jesteÅmy w Dockerze, tylko testujemy lokalnie (uÅ¼ywamy localhost i portu 5435)
        try:
            conn = psycopg2.connect(
                host="localhost",
                port="5435",     # TwÃ³j port ZEWNÄTRZNY
                database="redline_db",
                user="redline_user",
                password="redline_pass"
            )
            return conn
        except Exception as e_local:
            print(f"â BÅÄD BAZY DANYCH: Nie udaÅo siÄ poÅÄczyÄ ani przez Docker, ani localhost.")
            print(f"   Docker error: {e_docker}")
            print(f"   Local error: {e_local}")
            raise e_local

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    user_msg = request.message.lower()
    
    # 1. KONFIGURACJA POÅÄCZENIA Z LM STUDIO (Host Windows widziany z Dockera)
    LLM_URL = "http://host.docker.internal:1234/v1/chat/completions"
    HEADERS = {"Content-Type": "application/json"}
    
    # DomyÅlna odpowiedÅº awaryjna
    bot_response = "â ï¸ System AI offline. Nie mogÄ poÅÄczyÄ siÄ z modelem."
    
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
    
    # 3. LOGIKA BAZY DANYCH (Zabezpieczenie przed halucynacjÄ)
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
                # BARDZO WAÅ»NE: Wymuszamy format odpowiedzi w System Prompt
                price_context = f"AKTUALNA CENA Z BAZY DANYCH: {detected_symbol} = {price:.2f} USDT."
                is_price_query = True
            else:
                price_context = "BRAK DANYCH W BAZIE. Nie zmyÅlaj ceny."
        except Exception as e:
            price_context = f"BÅÄD BAZY: {str(e)}. Nie zmyÅlaj ceny."
    
    # 4. TURBO PROMPT (Dla szybkoÅci)
    if is_price_query:
        system_prompt = (
            f"JesteÅ Redline. {price_context} "
            "Podaj tÄ cenÄ uÅ¼ytkownikowi i dodaj krÃ³tki, cwaniacki komentarz po polsku. "
            "Maksymalnie 2 zdania."
        )
    else:
        system_prompt = (
            "JesteÅ Redline. MÃ³wisz krÃ³tko, zwiÄÅºle i po polsku. "
            "Nie tÅumacz siÄ. Odpowiadaj jak ekspert."
        )

    # 5. WYSÅANIE ZAPYTANIA DO AI
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ],
        "temperature": 0.3,   # MaÅa kreatywnoÅÄ = szybsze myÅlenie
        "max_tokens": 60      # LIMIT SÅÃW! Ucinamy go szybko, Å¼eby nie muliÅ.
    }

    try:
        # Timeout 20s Å¼eby nie blokowaÄ interfejsu
        response = requests.post(LLM_URL, headers=HEADERS, json=payload, timeout=200)
        
        if response.status_code == 200:
            data = response.json()
            bot_response = data['choices'][0]['message']['content']
        else:
            bot_response = f"â BÅÄd LLM ({response.status_code}): {response.text}"
            
    except requests.exceptions.ConnectionError:
        # FALLBACK: JeÅli LM Studio nie dziaÅa, zwrÃ³Ä chociaÅ¼ cenÄ z bazy
        if is_price_query:
            bot_response = f"{price_context} (â ï¸ ModuÅ AI Offline - uruchom LM Studio 'Server')."
        else:
            bot_response = "ð BÅÄd poÅÄczenia z mÃ³zgiem AI. Upewnij siÄ, Å¼e serwer LM Studio jest aktywny."
    except Exception as e:
        bot_response = f"â Krytyczny bÅÄd: {str(e)}"

    return {"response": bot_response}


# --- REGISTRATION & ADMIN APPROVAL ENDPOINTS ---

class RegistrationData(BaseModel):
    fullName: str
    email: str
    password: str
    experience: Optional[str] = None
    portfolioSize: Optional[str] = None
    tradingStyle: Optional[str] = None
    riskTolerance: Optional[str] = None
    exchanges: List[str] = []
    tradingCoins: Optional[str] = None
    hearAbout: Optional[str] = None
    referralCode: Optional[str] = None

class ApprovalData(BaseModel):
    adminNotes: Optional[str] = None

class RejectionData(BaseModel):
    adminNotes: Optional[str] = None
    reason: Optional[str] = None

@app.post("/api/register")
async def register_user(data: RegistrationData):
    """
    Register a new user - stores application in pending_registrations table
    """
    try:
        import hashlib
        import json
        
        # Hash password
        password_hash = hashlib.sha256(data.password.encode()).hexdigest()
        
        # Convert exchanges list to JSON string for SQLite
        exchanges_str = json.dumps(data.exchanges)
        
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        # Check if email already exists
        cursor.execute("SELECT id FROM pending_registrations WHERE email = ?", (data.email,))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Insert into pending_registrations
        cursor.execute("""
            INSERT INTO pending_registrations 
            (full_name, email, password_hash, experience, portfolio_size, trading_style, 
             risk_tolerance, exchanges, trading_coins, hear_about, referral_code, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            data.fullName,
            data.email,
            password_hash,
            data.experience,
            data.portfolioSize,
            data.tradingStyle,
            data.riskTolerance,
            exchanges_str,
            data.tradingCoins,
            data.hearAbout,
            data.referralCode
        ))
        
        application_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"â New registration application from {data.email} (ID: {application_id})")
        
        return {
            "success": True,
            "message": "Application submitted successfully",
            "applicationId": application_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"â Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.get("/api/admin/applications")
async def get_applications(status: Optional[str] = None):
    """
    Get all registration applications (admin only)
    """
    try:
        import json
        
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT * FROM pending_registrations 
                WHERE status = ? 
                ORDER BY submitted_at DESC
            """, (status,))
        else:
            cursor.execute("""
                SELECT * FROM pending_registrations 
                ORDER BY submitted_at DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        applications = []
        for row in rows:
            app = dict(row)
            # Parse exchanges JSON string back to list
            if app['exchanges']:
                try:
                    app['exchanges'] = json.loads(app['exchanges'])
                except:
                    app['exchanges'] = []
            else:
                app['exchanges'] = []
            applications.append(app)
        
        return {"applications": applications}
        
    except Exception as e:
        logger.error(f"â Error fetching applications: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/applications/{application_id}")
async def get_application(application_id: int):
    """
    Get single application details (admin only)
    """
    try:
        import json
        
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM pending_registrations WHERE id = ?", (application_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Application not found")
        
        app = dict(row)
        # Parse exchanges JSON
        if app['exchanges']:
            try:
                app['exchanges'] = json.loads(app['exchanges'])
            except:
                app['exchanges'] = []
        else:
            app['exchanges'] = []
        
        return app
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"â Error fetching application: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/applications/{application_id}/approve")
async def approve_application(application_id: int, data: ApprovalData):
    """
    Approve a registration application and create user account (admin only)
    """
    try:
        from datetime import datetime
        
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        # Get application data
        cursor.execute("SELECT * FROM pending_registrations WHERE id = ?", (application_id,))
        app = cursor.fetchone()
        
        if not app:
            conn.close()
            raise HTTPException(status_code=404, detail="Application not found")
        
        if app['status'] != 'pending':
            conn.close()
            raise HTTPException(status_code=400, detail="Application already processed")
        
        # Create user account
        username = app['email'].split('@')[0]  # Use email prefix as username
        
        cursor.execute("""
            INSERT INTO users (username, hash, contact, role, created_at)
            VALUES (?, ?, ?, 'USER', ?)
        """, (username, app['password_hash'], app['email'], datetime.now()))
        
        # Update application status
        cursor.execute("""
            UPDATE pending_registrations 
            SET status = 'approved', 
                reviewed_at = ?,
                reviewed_by = 'admin',
                admin_notes = ?
            WHERE id = ?
        """, (datetime.now(), data.adminNotes, application_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"â Application {application_id} approved - User '{username}' created")
        
        return {
            "success": True,
            "message": "Application approved and user account created",
            "username": username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"â Error approving application: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/applications/{application_id}/reject")
async def reject_application(application_id: int, data: RejectionData):
    """
    Reject a registration application (admin only)
    """
    try:
        from datetime import datetime
        
        conn = Database.get_connection()
        cursor = conn.cursor()
        
        # Check if application exists
        cursor.execute("SELECT status FROM pending_registrations WHERE id = ?", (application_id,))
        app = cursor.fetchone()
        
        if not app:
            conn.close()
            raise HTTPException(status_code=404, detail="Application not found")
        
        if app['status'] != 'pending':
            conn.close()
            raise HTTPException(status_code=400, detail="Application already processed")
        
        # Update application status
        notes = data.adminNotes or ""
        if data.reason:
            notes = f"Reason: {data.reason}\n{notes}"
        
        cursor.execute("""
            UPDATE pending_registrations 
            SET status = 'rejected',
                reviewed_at = ?,
                reviewed_by = 'admin',
                admin_notes = ?
            WHERE id = ?
        """, (datetime.now(), notes, application_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"â Application {application_id} rejected")
        
        return {
            "success": True,
            "message": "Application rejected"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"â Error rejecting application: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================

# ADMIN DASHBOARD ENDPOINTS

# ==========================================



from typing import List, Optional

from datetime import datetime, timedelta

import json

import hashlib



# --- PYDANTIC MODELS ---



class UserUpdate(BaseModel):

    role: Optional[str] = None

    contact: Optional[str] = None

    risk_limit: Optional[float] = None

    trading_enabled: Optional[bool] = None

    api_key: Optional[str] = None

    api_secret: Optional[str] = None

    exchange: Optional[str] = None

    notes: Optional[str] = None

    status: Optional[str] = None



class UserCreate(BaseModel):

    username: str

    password: str

    role: str = 'VIEWER'

    contact: Optional[str] = None

    risk_limit: float = 1000.0

    trading_enabled: bool = False

    exchange: str = 'BINANCE'



class EmailSettings(BaseModel):

    smtp_server: Optional[str] = None

    smtp_port: Optional[int] = None

    smtp_username: Optional[str] = None

    smtp_password: Optional[str] = None

    from_email: Optional[str] = None

    auto_send_enabled: bool = False



class EmailSend(BaseModel):

    to_email: str

    subject: str

    body: str

    template_name: Optional[str] = None



# --- HELPER FUNCTIONS ---



def log_audit(admin_username: str, action_type: str, target: str, details: dict = None, ip: str = None):

    """Log admin action to audit log"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        cursor.execute("""

            INSERT INTO audit_log (admin_username, action_type, target, details, ip_address)

            VALUES (?, ?, ?, ?, ?)

        """, (admin_username, action_type, target, json.dumps(details) if details else None, ip))

        conn.commit()

        conn.close()

        logger.info(f"z 9  Audit: {admin_username} -> {action_type} -> {target}")

    except Exception as e:

        logger.error(f"âeZ Audit log error: {str(e)}")



# ==========================================

# USER MANAGEMENT ENDPOINTS

# ==========================================



@app.get("/api/admin/users")

async def get_all_users(

    search: Optional[str] = None,

    role: Optional[str] = None,

    status: Optional[str] = None

):

    """Get all users with optional filters"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        query = "SELECT * FROM users WHERE 1=1"

        params = []

        

        if search:

            query += " AND (username LIKE ? OR contact LIKE ?)"

            params.extend([f"%{search}%", f"%{search}%"])

        

        if role:

            query += " AND role = ?"

            params.append(role)

        

        if status:

            query += " AND status = ?"

            params.append(status)

        

        query += " ORDER BY created_at DESC"

        

        cursor.execute(query, params)

        rows = cursor.fetchall()

        conn.close()

        

        users = []

        for row in rows:

            user = dict(row)

            # Don't send password hash to frontend

            user.pop('hash', None)

            users.append(user)

        

        return {"users": users}

        

    except Exception as e:

        logger.error(f"âeZ Error fetching users: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/admin/users/{username}")

async def get_user(username: str):

    """Get single user details"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))

        row = cursor.fetchone()

        conn.close()

        

        if not row:

            raise HTTPException(status_code=404, detail="User not found")

        

        user = dict(row)

        user.pop('hash', None)  # Don't send password hash

        

        return user

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error fetching user: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.put("/api/admin/users/{username}")

async def update_user(username: str, data: UserUpdate):

    """Update user details"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        # Check if user exists

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))

        if not cursor.fetchone():

            conn.close()

            raise HTTPException(status_code=404, detail="User not found")

        

        # Build update query

        updates = []

        params = []

        

        if data.role is not None:

            updates.append("role = ?")

            params.append(data.role)

        if data.contact is not None:

            updates.append("contact = ?")

            params.append(data.contact)

        if data.risk_limit is not None:

            updates.append("risk_limit = ?")

            params.append(data.risk_limit)

        if data.trading_enabled is not None:

            updates.append("trading_enabled = ?")

            params.append(data.trading_enabled)

        if data.api_key is not None:

            updates.append("api_key = ?")

            params.append(data.api_key)

        if data.api_secret is not None:

            updates.append("api_secret = ?")

            params.append(data.api_secret)

        if data.exchange is not None:

            updates.append("exchange = ?")

            params.append(data.exchange)

        if data.notes is not None:

            updates.append("notes = ?")

            params.append(data.notes)

        if data.status is not None:

            updates.append("status = ?")

            params.append(data.status)

        

        if not updates:

            conn.close()

            return {"success": True, "message": "No changes"}

        

        params.append(username)

        query = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"

        

        cursor.execute(query, params)

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "update_user", username, data.dict(exclude_none=True))

        

        logger.info(f"â[&  User {username} updated")

        return {"success": True, "message": "User updated"}

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error updating user: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/api/admin/users/{username}")

async def delete_user(username: str):

    """Delete user"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("DELETE FROM users WHERE username = ?", (username,))

        

        if cursor.rowcount == 0:

            conn.close()

            raise HTTPException(status_code=404, detail="User not found")

        

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "delete_user", username)

        

        logger.info(f"z  ¸y User {username} deleted")

        return {"success": True, "message": "User deleted"}

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error deleting user: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/admin/users/{username}/block")

async def block_user(username: str):

    """Block user account"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("UPDATE users SET status = 'blocked' WHERE username = ?", (username,))

        

        if cursor.rowcount == 0:

            conn.close()

            raise HTTPException(status_code=404, detail="User not found")

        

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "block_user", username)

        

        logger.info(f"z   User {username} blocked")

        return {"success": True, "message": "User blocked"}

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error blocking user: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/admin/users/{username}/unblock")

async def unblock_user(username: str):

    """Unblock user account"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("UPDATE users SET status = 'active' WHERE username = ?", (username,))

        

        if cursor.rowcount == 0:

            conn.close()

            raise HTTPException(status_code=404, detail="User not found")

        

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "unblock_user", username)

        

        logger.info(f"z   User {username} unblocked")

        return {"success": True, "message": "User unblocked"}

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error unblocking user: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/admin/users/{username}/reset-password")

async def reset_password(username: str):

    """Reset user password to temporary password"""

    try:

        import secrets

        

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        # Generate temporary password

        temp_password = secrets.token_urlsafe(12)

        password_hash = hashlib.sha256(temp_password.encode()).hexdigest()

        

        cursor.execute("UPDATE users SET hash = ? WHERE username = ?", (password_hash, username))

        

        if cursor.rowcount == 0:

            conn.close()

            raise HTTPException(status_code=404, detail="User not found")

        

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "reset_password", username)

        

        logger.info(f"z   Password reset for {username}")

        return {

            "success": True,

            "message": "Password reset",

            "temp_password": temp_password

        }

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error resetting password: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/admin/users")

async def create_user(data: UserCreate):

    """Create new user"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        # Check if username exists

        cursor.execute("SELECT username FROM users WHERE username = ?", (data.username,))

        if cursor.fetchone():

            conn.close()

            raise HTTPException(status_code=400, detail="Username already exists")

        

        # Hash password

        password_hash = hashlib.sha256(data.password.encode()).hexdigest()

        

        cursor.execute("""

            INSERT INTO users (username, hash, role, contact, risk_limit, trading_enabled, exchange, status)

            VALUES (?, ?, ?, ?, ?, ?, ?, 'active')

        """, (

            data.username,

            password_hash,

            data.role,

            data.contact,

            data.risk_limit,

            data.trading_enabled,

            data.exchange

        ))

        

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "create_user", data.username, {"role": data.role})

        

        logger.info(f"â[&  User {data.username} created")

        return {"success": True, "message": "User created", "username": data.username}

        

    except HTTPException:

        raise

    except Exception as e:

        logger.error(f"âeZ Error creating user: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



# ==========================================

# STATISTICS ENDPOINTS

# ==========================================



@app.get("/api/admin/stats/overview")

async def get_stats_overview():

    """Get overview statistics"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        # Total users

        cursor.execute("SELECT COUNT(*) FROM users")

        total_users = cursor.fetchone()[0]

        

        # Pending applications

        cursor.execute("SELECT COUNT(*) FROM pending_registrations WHERE status = 'pending'")

        pending_apps = cursor.fetchone()[0]

        

        # Approval rate

        cursor.execute("""

            SELECT 

                COUNT(CASE WHEN status = 'approved' THEN 1 END) * 100.0 / 

                NULLIF(COUNT(CASE WHEN status IN ('approved', 'rejected') THEN 1 END), 0)

            FROM pending_registrations

        """)

        approval_rate = cursor.fetchone()[0] or 0

        

        # Active today (users created today)

        cursor.execute("""

            SELECT COUNT(*) FROM users 

            WHERE DATE(created_at) = DATE('now')

        """)

        active_today = cursor.fetchone()[0]

        

        # Users by role

        cursor.execute("""

            SELECT role, COUNT(*) as count 

            FROM users 

            GROUP BY role

        """)

        users_by_role = {row[0]: row[1] for row in cursor.fetchall()}

        

        conn.close()

        

        return {

            "total_users": total_users,

            "pending_applications": pending_apps,

            "approval_rate": round(approval_rate, 1),

            "active_today": active_today,

            "users_by_role": users_by_role

        }

        

    except Exception as e:

        logger.error(f"âeZ Error fetching stats: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/admin/stats/user-growth")

async def get_user_growth():

    """Get user growth data for last 30 days"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("""

            SELECT DATE(created_at) as date, COUNT(*) as count

            FROM users

            WHERE created_at >= DATE('now', '-30 days')

            GROUP BY DATE(created_at)

            ORDER BY date ASC

        """)

        

        rows = cursor.fetchall()

        conn.close()

        

        data = [{"date": row[0], "count": row[1]} for row in rows]

        

        return {"growth_data": data}

        

    except Exception as e:

        logger.error(f"âeZ Error fetching user growth: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/admin/stats/applications")

async def get_application_stats():

    """Get application statistics"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("""

            SELECT status, COUNT(*) as count

            FROM pending_registrations

            GROUP BY status

        """)

        

        rows = cursor.fetchall()

        conn.close()

        

        stats = {row[0]: row[1] for row in rows}

        

        return {"application_stats": stats}

        

    except Exception as e:

        logger.error(f"âeZ Error fetching application stats: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



# ==========================================

# EMAIL SYSTEM ENDPOINTS

# ==========================================



@app.get("/api/admin/email/settings")

async def get_email_settings():

    """Get email settings"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("SELECT * FROM email_settings WHERE id = 1")

        row = cursor.fetchone()

        conn.close()

        

        if not row:

            return {

                "smtp_server": "",

                "smtp_port": 587,

                "smtp_username": "",

                "from_email": "",

                "auto_send_enabled": False

            }

        

        settings = dict(row)

        # Don't send password to frontend

        settings.pop('smtp_password', None)

        settings.pop('id', None)

        

        return settings

        

    except Exception as e:

        logger.error(f"âeZ Error fetching email settings: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.put("/api/admin/email/settings")

async def update_email_settings(data: EmailSettings):

    """Update email settings"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        # Check if settings exist

        cursor.execute("SELECT id FROM email_settings WHERE id = 1")

        exists = cursor.fetchone()

        

        if exists:

            # Update

            updates = []

            params = []

            

            if data.smtp_server is not None:

                updates.append("smtp_server = ?")

                params.append(data.smtp_server)

            if data.smtp_port is not None:

                updates.append("smtp_port = ?")

                params.append(data.smtp_port)

            if data.smtp_username is not None:

                updates.append("smtp_username = ?")

                params.append(data.smtp_username)

            if data.smtp_password is not None:

                updates.append("smtp_password = ?")

                params.append(data.smtp_password)

            if data.from_email is not None:

                updates.append("from_email = ?")

                params.append(data.from_email)

            

            updates.append("auto_send_enabled = ?")

            params.append(data.auto_send_enabled)

            

            query = f"UPDATE email_settings SET {', '.join(updates)} WHERE id = 1"

            cursor.execute(query, params)

        else:

            # Insert

            cursor.execute("""

                INSERT INTO email_settings (id, smtp_server, smtp_port, smtp_username, smtp_password, from_email, auto_send_enabled)

                VALUES (1, ?, ?, ?, ?, ?, ?)

            """, (

                data.smtp_server,

                data.smtp_port,

                data.smtp_username,

                data.smtp_password,

                data.from_email,

                data.auto_send_enabled

            ))

        

        conn.commit()

        conn.close()

        

        # Log audit

        log_audit("admin", "update_email_settings", "email_settings")

        

        logger.info("â[&  Email settings updated")

        return {"success": True, "message": "Email settings updated"}

        

    except Exception as e:

        logger.error(f"âeZ Error updating email settings: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/admin/email/send")

async def send_email(data: EmailSend):

    """Send email (placeholder - requires SMTP configuration)"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        # Log email

        cursor.execute("""

            INSERT INTO email_log (to_email, subject, template_name, status)

            VALUES (?, ?, ?, 'queued')

        """, (data.to_email, data.subject, data.template_name))

        

        conn.commit()

        conn.close()

        

        # TODO: Implement actual email sending with SMTP

        logger.info(f"z § Email queued to {data.to_email}")

        

        return {

            "success": True,

            "message": "Email queued (SMTP not configured yet)"

        }

        

    except Exception as e:

        logger.error(f"âeZ Error sending email: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/admin/email/log")

async def get_email_log(limit: int = 50):

    """Get email log"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("""

            SELECT * FROM email_log 

            ORDER BY sent_at DESC 

            LIMIT ?

        """, (limit,))

        

        rows = cursor.fetchall()

        conn.close()

        

        logs = [dict(row) for row in rows]

        

        return {"email_log": logs}

        

    except Exception as e:

        logger.error(f"âeZ Error fetching email log: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



# ==========================================

# AUDIT LOG ENDPOINTS

# ==========================================



@app.get("/api/admin/audit-log")

async def get_audit_log(

    limit: int = 100,

    admin: Optional[str] = None,

    action: Optional[str] = None,

    target: Optional[str] = None

):

    """Get audit log with filters"""

    try:

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        query = "SELECT * FROM audit_log WHERE 1=1"

        params = []

        

        if admin:

            query += " AND admin_username = ?"

            params.append(admin)

        

        if action:

            query += " AND action_type = ?"

            params.append(action)

        

        if target:

            query += " AND target LIKE ?"

            params.append(f"%{target}%")

        

        query += " ORDER BY timestamp DESC LIMIT ?"

        params.append(limit)

        

        cursor.execute(query, params)

        rows = cursor.fetchall()

        conn.close()

        

        logs = []

        for row in rows:

            log = dict(row)

            # Parse JSON details

            if log.get('details'):

                try:

                    log['details'] = json.loads(log['details'])

                except:

                    pass

            logs.append(log)

        

        return {"audit_log": logs}

        

    except Exception as e:

        logger.error(f"âeZ Error fetching audit log: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/admin/audit-log/export")

async def export_audit_log():

    """Export audit log to CSV"""

    try:

        import csv

        from io import StringIO

        

        conn = Database.get_connection()

        cursor = conn.cursor()

        

        cursor.execute("SELECT * FROM audit_log ORDER BY timestamp DESC")

        rows = cursor.fetchall()

        conn.close()

        

        # Create CSV

        output = StringIO()

        writer = csv.writer(output)

        

        # Header

        writer.writerow(['ID', 'Timestamp', 'Admin', 'Action', 'Target', 'Details', 'IP'])

        

        # Data

        for row in rows:

            writer.writerow([

                row['id'],

                row['timestamp'],

                row['admin_username'],

                row['action_type'],

                row['target'],

                row['details'],

                row['ip_address']

            ])

        

        csv_content = output.getvalue()

        

        return {

            "success": True,

            "csv": csv_content,

            "filename": f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        }

        

    except Exception as e:

        logger.error(f"âeZ Error exporting audit log: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))

