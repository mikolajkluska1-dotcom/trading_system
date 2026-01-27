# chat_endpoints.py - Chat API Endpoints
"""
Chat endpoints for multi-model AI conversation:
- POST /api/chat/send - Send message to AI model
- GET /api/chat/history - Get conversation history
- DELETE /api/chat/clear - Clear conversation
- GET /api/chat/models - List available models
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# Import model handlers
from chat_models import get_model_response, get_all_models, MODELS

# Database connection
def get_db():
    """Get database connection"""
    conn = sqlite3.connect('../../Database/assets/trading_system.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize chat_messages table
def init_chat_db():
    """Create chat_messages table if not exists"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                model TEXT,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing chat DB: {e}")


# Initialize on import
init_chat_db()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatMessage(BaseModel):
    model: str
    message: str
    conversation_id: Optional[str] = None


# ============================================================================
# ENDPOINT 1: Send Message
# ============================================================================

def send_chat_message(data: dict):
    """
    POST /api/chat/send
    Send message to AI model and get response
    """
    try:
        model_id = data.get('model', 'general')
        message = data.get('message', '')
        conversation_id = data.get('conversation_id') or str(uuid.uuid4())
        
        if not message:
            return {"success": False, "error": "Message cannot be empty"}
        
        if model_id not in MODELS:
            return {"success": False, "error": f"Invalid model: {model_id}"}
        
        # Save user message
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (conversation_id, role, model, content)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, 'user', None, message))
        
        # Get AI response
        response_content = get_model_response(model_id, message)
        
        # Save AI response
        cursor.execute("""
            INSERT INTO chat_messages (conversation_id, role, model, content)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, 'assistant', model_id, response_content))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "response": response_content,
            "model": model_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 2: Get Conversation History
# ============================================================================

def get_chat_history(conversation_id: str):
    """
    GET /api/chat/history?conversation_id=xxx
    Get all messages in a conversation
    """
    try:
        if not conversation_id:
            return {"success": False, "error": "conversation_id required"}
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT role, model, content, timestamp
            FROM chat_messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (conversation_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "role": row['role'],
                "model": row['model'],
                "content": row['content'],
                "timestamp": row['timestamp']
            })
        
        conn.close()
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "messages": messages
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 3: Clear Conversation
# ============================================================================

def clear_chat_history(conversation_id: str):
    """
    DELETE /api/chat/clear?conversation_id=xxx
    Clear all messages in a conversation
    """
    try:
        if not conversation_id:
            return {"success": False, "error": "conversation_id required"}
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM chat_messages
            WHERE conversation_id = ?
        """, (conversation_id,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "deleted_messages": deleted
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 4: List Available Models
# ============================================================================

def get_available_models():
    """
    GET /api/chat/models
    Get list of all available AI models
    """
    try:
        models = get_all_models()
        
        return {
            "success": True,
            "models": models
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
