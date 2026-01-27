# chat_api.py - FastAPI wrapper for chat endpoints
"""
FastAPI application serving chat endpoints
Run with: uvicorn chat_api:app --host 0.0.0.0 --port 8002
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import endpoint functions
from chat_endpoints import (
    send_chat_message,
    get_chat_history,
    clear_chat_history,
    get_available_models
)

# Create FastAPI app
app = FastAPI(title="AI Chat API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/api/chat/send")
async def send_message(data: dict):
    """Send message to AI model"""
    return send_chat_message(data)


@app.get("/api/chat/history")
async def get_history(conversation_id: str = Query(...)):
    """Get conversation history"""
    return get_chat_history(conversation_id)


@app.delete("/api/chat/clear")
async def clear_history(conversation_id: str = Query(...)):
    """Clear conversation history"""
    return clear_chat_history(conversation_id)


@app.get("/api/chat/models")
async def list_models():
    """List available AI models"""
    return get_available_models()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chat-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
