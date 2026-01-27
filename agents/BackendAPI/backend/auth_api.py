# auth_api.py - Authentication API Endpoints
"""
Authentication endpoints for login and session management
Run with: uvicorn auth_api:app --host 0.0.0.0 --port 8003
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Import UserManager
from agents.BackendAPI.security.user_manager import UserManager

# Create FastAPI app
app = FastAPI(title="Auth API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    contact: str = ""

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """
    Login endpoint
    Returns user role and username if successful
    """
    try:
        role = UserManager.verify_login(req.username, req.password)
        
        if role:
            return {
                "success": True,
                "username": req.username,
                "role": role,
                "message": "Login successful"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """
    Register new user (request account)
    """
    try:
        success, message = UserManager.request_account(
            req.username,
            req.password,
            req.contact
        )
        
        if success:
            return {
                "success": True,
                "message": message
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Register error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/auth/me")
async def get_current_user(username: str):
    """
    Get current user info
    """
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
        else:
            raise HTTPException(status_code=404, detail="User not found")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "auth-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
