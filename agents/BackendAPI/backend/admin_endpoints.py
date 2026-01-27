
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
        logger.info(f"📋 Audit: {admin_username} -> {action_type} -> {target}")
    except Exception as e:
        logger.error(f"❌ Audit log error: {str(e)}")

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
        logger.error(f"❌ Error fetching users: {str(e)}")
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
        logger.error(f"❌ Error fetching user: {str(e)}")
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
        
        logger.info(f"✅ User {username} updated")
        return {"success": True, "message": "User updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating user: {str(e)}")
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
        
        logger.info(f"🗑️ User {username} deleted")
        return {"success": True, "message": "User deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting user: {str(e)}")
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
        
        logger.info(f"🔒 User {username} blocked")
        return {"success": True, "message": "User blocked"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error blocking user: {str(e)}")
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
        
        logger.info(f"🔓 User {username} unblocked")
        return {"success": True, "message": "User unblocked"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error unblocking user: {str(e)}")
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
        
        logger.info(f"🔑 Password reset for {username}")
        return {
            "success": True,
            "message": "Password reset",
            "temp_password": temp_password
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resetting password: {str(e)}")
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
        
        logger.info(f"✅ User {data.username} created")
        return {"success": True, "message": "User created", "username": data.username}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating user: {str(e)}")
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
        logger.error(f"❌ Error fetching stats: {str(e)}")
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
        logger.error(f"❌ Error fetching user growth: {str(e)}")
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
        logger.error(f"❌ Error fetching application stats: {str(e)}")
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
        logger.error(f"❌ Error fetching email settings: {str(e)}")
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
        
        logger.info("✅ Email settings updated")
        return {"success": True, "message": "Email settings updated"}
        
    except Exception as e:
        logger.error(f"❌ Error updating email settings: {str(e)}")
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
        logger.info(f"📧 Email queued to {data.to_email}")
        
        return {
            "success": True,
            "message": "Email queued (SMTP not configured yet)"
        }
        
    except Exception as e:
        logger.error(f"❌ Error sending email: {str(e)}")
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
        logger.error(f"❌ Error fetching email log: {str(e)}")
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
        logger.error(f"❌ Error fetching audit log: {str(e)}")
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
        logger.error(f"❌ Error exporting audit log: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
