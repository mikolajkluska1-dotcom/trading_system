import json
import os
import hashlib

# 1. KONFIGURACJA ODZYSKIWANIA
ADMIN_PASS = "Cyber_Fortress_X1"
OVERRIDE_PASS = "Redline_OMEGA_99!"
SALT = "X9vL2_REDLINE_STATIC_SALT"

def fix_users_db():
    print(f"🔧 [1/2] Naprawianie bazy użytkowników (assets/users_db.json)...")
    
    # Hashowanie hasła admina (Zwykłe SHA-256, bez soli - tak jak w ui/auth.py)
    admin_hash = hashlib.sha256(ADMIN_PASS.encode()).hexdigest()
    
    db_structure = {
        "active": {
            "admin": {
                "hash": admin_hash,
                "role": "ROOT",
                "allowed_ips": [] # Pusta lista wymusi Override przy nowym IP
            }
        },
        "pending": {}
    }
    
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    with open(os.path.join("assets", "users_db.json"), "w") as f:
        json.dump(db_structure, f, indent=4)
        
    print(f"✅ Baza naprawiona. Login: admin | Hasło: {ADMIN_PASS}")

def fix_config_hash():
    print(f"🔧 [2/2] Generowanie Hasha dla Configu...")
    
    # Hashowanie Override Code (Hasło + Sól - tak jak w ui/auth.py sekcja Override)
    combined = OVERRIDE_PASS + SALT
    override_hash = hashlib.sha256(combined.encode()).hexdigest()
    
    print("\n⚠️ SKOPIUJ PONIŻSZĄ LINIĘ DO PLIKU core/config.py:")
    print("-" * 60)
    print(f'SECURITY_OVERRIDE_HASH = "{override_hash}"')
    print("-" * 60)

if __name__ == "__main__":
    fix_users_db()
    fix_config_hash()