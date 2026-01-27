from agents.Database.core.db import Database
from agents.BackendAPI.security.vault import Vault
import os

class WalletManager:
    """Zarządca stanu portfela (SQLite Powered)"""

    @staticmethod
    def migrate_if_needed():
        """Migracja z zaszyfrowanego pliku Vault do SQLite."""
        enc_file = os.path.join("assets", "wallet.enc")
        if not os.path.exists(enc_file):
            return

        try:
            # Vault.load_wallet() odszyfruje dane
            data = Vault.load_wallet()
            if data.get('ERROR') or data.get('LOCKED'):
                return

            conn = Database.get_connection()
            # 1. Update Balance
            conn.execute("UPDATE wallet SET balance = ? WHERE id = 1", (data.get('balance', 1000.0),))

            # 2. Add Assets
            for asset in data.get('assets', []):
                conn.execute("""
                    INSERT OR REPLACE INTO assets (symbol, amount, entry_price)
                    VALUES (?, ?, ?)
                """, (asset['sym'], asset['size'], asset.get('entry', 0)))

            # 3. History
            for trade in data.get('history', []):
                conn.execute("""
                    INSERT INTO trade_history (symbol, side, price, amount, cost, status, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.get('symbol'), trade.get('side'), trade.get('price'),
                    trade.get('amount'), trade.get('cost'), trade.get('status', 'FILLED'),
                    trade.get('ts')
                ))

            conn.commit()
            conn.close()

            # Zmiana nazwy starego pliku
            os.rename(enc_file, enc_file + ".bak")
            print(f" ✅ Migrated wallet to SQLite from {enc_file}")
        except Exception as e:
            print(f" ❌ Wallet Migration Error: {e}")

    @staticmethod
    def get_wallet_data():
        WalletManager.migrate_if_needed()
        conn = Database.get_connection()
        balance = conn.execute("SELECT balance FROM wallet WHERE id = 1").fetchone()['balance']
        assets = [dict(a) for a in conn.execute("SELECT symbol as sym, amount as size, entry_price as entry FROM assets")]
        history = [dict(h) for h in conn.execute("SELECT * FROM trade_history ORDER BY ts DESC LIMIT 50")]
        conn.close()

        return {
            "balance": balance,
            "assets": assets,
            "history": history
        }

    @staticmethod
    def get_balance():
        conn = Database.get_connection()
        res = conn.execute("SELECT balance FROM wallet WHERE id = 1").fetchone()
        conn.close()
        return res['balance'] if res else 0.0

    @staticmethod
    def get_assets():
        conn = Database.get_connection()
        assets = [dict(a) for a in conn.execute("SELECT symbol as sym, amount as size, entry_price as entry FROM assets WHERE amount > 0")]
        conn.close()
        return assets

    @staticmethod
    def update_balance(new_balance):
        conn = Database.get_connection()
        conn.execute("UPDATE wallet SET balance = ? WHERE id = 1", (new_balance,))
        conn.commit()
        conn.close()

    @staticmethod
    def record_trade(symbol, side, price, amount, cost, signal_id=None, reason=None):
        conn = Database.get_connection()
        conn.execute("""
            INSERT INTO trade_history (symbol, side, price, amount, cost, signal_id, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, side, price, amount, cost, signal_id, reason))

        # Update assets
        if side == "BUY":
            conn.execute("""
                INSERT INTO assets (symbol, amount, entry_price)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                amount = amount + EXCLUDED.amount,
                entry_price = (entry_price * amount + EXCLUDED.entry_price * EXCLUDED.amount) / (amount + EXCLUDED.amount)
            """, (symbol, amount, price))
        else: # SELL
            conn.execute("UPDATE assets SET amount = amount - ? WHERE symbol = ?", (amount, symbol))
            conn.execute("DELETE FROM assets WHERE amount <= 0")

        conn.commit()
        conn.close()
