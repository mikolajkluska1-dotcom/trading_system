from security.user_manager import UserManager
from core.db import Database
import logging

# Setup logging to see what happens
logging.basicConfig(level=logging.INFO)

print("--- FORCING ADMIN PASSWORD RESET (VIA APP LOGIC) ---")

try:
    # 1. Force logic to run
    UserManager.migrate_if_needed()
    
    # 2. Connect
    print("Connecting to DB via Core...")
    conn = Database.get_connection()
    cursor = conn.cursor()
    
    # 3. Hash
    new_pass = "admin123"
    new_hash = UserManager.hash_password(new_pass)
    print(f"Generated Hash for '{new_pass}': {new_hash}")
    
    # 4. Update
    cursor.execute("DELETE FROM users WHERE username = 'admin'")
    cursor.execute("""
        INSERT INTO users (username, hash, role, contact, risk_limit, trading_enabled, exchange)
        VALUES (?, ?, 'ROOT', 'sysadmin', 1000000.0, 1, 'BINANCE')
    """, ('admin', new_hash))
    
    conn.commit()
    print("✅ User 'admin' recreated successfully.")
    
    # 5. Verify immediate
    check = conn.execute("SELECT hash FROM users WHERE username='admin'").fetchone()
    print(f"Verification Read: {check[0]}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
