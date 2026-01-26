"""
REDLINE ADMIN USER RESET SCRIPT
================================
Comprehensive script to reset the admin user with proper database initialization.

This script:
1. Initializes the database schema
2. Deletes any existing admin user
3. Creates a fresh admin user with correct credentials
4. Verifies the login works
5. Provides clear success/failure feedback

Usage:
    python backend/seed_admin_final.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("REDLINE ADMIN USER RESET")
    print("=" * 60)
    
    try:
        # Step 1: Initialize Database
        print("\n[1/5] Initializing database schema...")
        from core.db import Database
        Database.initialize()
        print("✅ Database schema initialized")
        
        # Step 2: Import UserManager
        print("\n[2/5] Loading user management system...")
        from security.user_manager import UserManager
        print("✅ UserManager loaded")
        
        # Step 3: Delete existing admin user (if any)
        print("\n[3/5] Removing existing admin user...")
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", ("admin",))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            print(f"✅ Deleted {deleted_count} existing admin user(s)")
        else:
            print("ℹ️  No existing admin user found")
        
        # Step 4: Create fresh admin user
        print("\n[4/5] Creating fresh admin user...")
        admin_password = "admin123"
        admin_hash = UserManager.hash_password(admin_password)
        
        # Verify hash matches expected value
        expected_hash = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
        if admin_hash != expected_hash:
            print(f"⚠️  WARNING: Generated hash doesn't match expected!")
            print(f"   Expected: {expected_hash}")
            print(f"   Got:      {admin_hash}")
            print("   Proceeding anyway...")
        
        conn = Database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users 
            (username, hash, role, contact, risk_limit, trading_enabled, exchange)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "admin",
            admin_hash,
            "ROOT",
            "sysadmin",
            1000000.0,
            1,  # trading_enabled = True
            "BINANCE"
        ))
        conn.commit()
        conn.close()
        print("✅ Admin user created successfully")
        print(f"   Username: admin")
        print(f"   Password: {admin_password}")
        print(f"   Role:     ROOT")
        print(f"   Hash:     {admin_hash}")
        
        # Step 5: Verify login works
        print("\n[5/5] Verifying login...")
        role = UserManager.verify_login("admin", admin_password)
        
        if role:
            print(f"✅ LOGIN VERIFICATION PASSED")
            print(f"   Verified role: {role}")
            print("\n" + "=" * 60)
            print("✅ ADMIN USER RESET SUCCESSFUL")
            print("=" * 60)
            print("\nYou can now login with:")
            print("  Username: admin")
            print("  Password: admin123")
            print("\nNext steps:")
            print("  1. Start the backend: uvicorn backend.main:app --reload")
            print("  2. Test login at: http://localhost:8000/api/auth/login")
            print("  3. Or use the frontend at: http://localhost:3000/login")
            return 0
        else:
            print("❌ LOGIN VERIFICATION FAILED")
            print("   The user was created but login doesn't work!")
            print("\nDebugging info:")
            
            # Check database
            conn = Database.get_connection()
            user = conn.execute("SELECT username, hash, role FROM users WHERE username = ?", ("admin",)).fetchone()
            conn.close()
            
            if user:
                print(f"   DB Username: {user['username']}")
                print(f"   DB Hash:     {user['hash']}")
                print(f"   DB Role:     {user['role']}")
                print(f"   Input Hash:  {admin_hash}")
                print(f"   Match:       {user['hash'] == admin_hash}")
            else:
                print("   User not found in database!")
            
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
