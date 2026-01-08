import requests
from .user_manager import UserManager

class NetworkSentinel:
    """
    Ochrona Sieciowa: IP Allowlist & Geo-Check.
    V3: Zabezpieczona przed 'Fallback Root Escalation'.
    """
    
    FALLBACK_IP = "127.0.0.1"
    
    @staticmethod
    def get_ip():
        """Pobiera IP z krótkim timeoutem."""
        try:
            ip = requests.get('https://api.ipify.org', timeout=3).text
            return ip
        except:
            return NetworkSentinel.FALLBACK_IP

    @staticmethod
    def check_access(user, current_ip):
        db = UserManager.load_db()
        
        if user not in db['active']: 
            return False
            
        user_data = db['active'][user]
        allowed_ips = user_data.get('allowed_ips', [])
        
        # FIRST BLOOD RULE: Auto-Add dla Roota
        if not allowed_ips and user_data['role'] == "ROOT":
            # 🛡️ SECURITY FIX: Nie pozwalamy na auto-add, jeśli IP to fallback!
            # Jeśli API padło, nie możemy ufać, że 127.0.0.1 to prawowity admin.
            if current_ip == NetworkSentinel.FALLBACK_IP:
                return False 
                
            NetworkSentinel.authorize_new_ip(user, current_ip)
            return True
            
        if current_ip in allowed_ips:
            return True
            
        return False

    @staticmethod
    def authorize_new_ip(user, current_ip):
        # Dodatkowe zabezpieczenie: nigdy nie dodawaj localhosta do bazy
        if current_ip == NetworkSentinel.FALLBACK_IP:
            return False
            
        db = UserManager.load_db()
        if user in db['active']:
            if 'allowed_ips' not in db['active'][user]:
                db['active'][user]['allowed_ips'] = []
            
            if current_ip not in db['active'][user]['allowed_ips']:
                db['active'][user]['allowed_ips'].append(current_ip)
                UserManager.save_db(db)
                return True
        return False