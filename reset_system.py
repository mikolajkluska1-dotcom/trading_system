import sqlite3
import os

DB_PATH = os.path.join("assets", "trading_system.db")

def reset_db():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Reset Wallet
        print("💰 Resetting Wallet to $1000...")
        cursor.execute("UPDATE wallet SET balance = 1000.0 WHERE id = 1")
        
        # 2. Clear Trade History
        print("deleted History...")
        cursor.execute("DELETE FROM trade_history")
        
        # 3. Clear Assets
        print("🗑️ Clearing Assets...")
        cursor.execute("DELETE FROM assets")

        conn.commit()
        print("✅ System Reset Complete. Ready for Night Run.")
        
    except Exception as e:
        print(f"❌ Error during reset: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    reset_db()
