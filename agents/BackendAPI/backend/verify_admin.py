import sqlite3
import sys
import os

# Change to project directory
os.chdir(r'c:\Users\user\Desktop\trading_system')

# Connect to database
conn = sqlite3.connect('assets/trading_system.db')
conn.row_factory = sqlite3.Row

# Query admin user
user = conn.execute('SELECT username, role, trading_enabled, hash FROM users WHERE username="admin"').fetchone()

if user:
    print(f"✅ Admin user found in database:")
    print(f"   Username: {user['username']}")
    print(f"   Role: {user['role']}")
    print(f"   Trading Enabled: {user['trading_enabled']}")
    print(f"   Hash: {user['hash']}")
else:
    print("❌ Admin user NOT found in database!")
    
conn.close()
