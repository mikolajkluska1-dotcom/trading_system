import hashlib
import sqlite3
import os

DB_FILE = os.path.join("assets", "trading_system.db")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def reset_admin():
    print(f"Connecting to {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    admin_hash = hash_password("admin123")
    print(f"Target Hash for 'admin123': {admin_hash}")
    
    # Check if admin exists
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    user = cursor.fetchone()
    
    if user:
        print(f"User 'admin' found. Current data: {user}")
        cursor.execute("UPDATE users SET hash = ?, role = 'ROOT', trading_enabled=1 WHERE username = 'admin'", (admin_hash,))
        print("Updated 'admin' password.")
    else:
        print("User 'admin' not found. Creating...")
        cursor.execute("""
            INSERT INTO users (username, hash, role, contact, risk_limit, trading_enabled, exchange)
            VALUES (?, ?, 'ROOT', 'sysadmin', 1000000.0, 1, 'BINANCE')
        """, ('admin', admin_hash))
        print("Created 'admin' user.")
        
    conn.commit()
    conn.close()
    print("✅ RESET COMPLETE. Try logging in with admin / admin123")

if __name__ == "__main__":
    try:
        reset_admin()
    except Exception as e:
        print(f"❌ Error: {e}")
