from security.vault import Vault

class WalletManager:
    """Zarządca stanu portfela (Read Ops)"""
    
    @staticmethod
    def get_wallet_data():
        return Vault.load_wallet()

    @staticmethod
    def get_balance():
        data = Vault.load_wallet()
        # Jeśli portfel jest zablokowany, zwracamy 0 dla bezpieczeństwa UI
        if data.get('LOCKED', False) or 'ERROR' in data:
            return 0.0
        return data.get('balance', 0.0)
        
    @staticmethod
    def get_assets():
        data = Vault.load_wallet()
        if data.get('LOCKED', False) or 'ERROR' in data:
            return []
        return data.get('assets', [])

    @staticmethod
    def save_wallet_data(data):
        Vault.save_wallet(data)