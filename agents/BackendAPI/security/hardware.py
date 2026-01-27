# ===========================================
# REDLINE V68 - HardwareSecurity V4.6.1
# ===========================================

import os, hashlib, secrets

class HardwareSecurity:
    KEY_FILENAME = "redline.key"
    EXPECTED_HASH = "replace_with_your_generated_hash"  # SHA256 hash of key content

    @staticmethod
    def verify_key(mount_path="E:/"):
        """Verifies a hardware key file by SHA256 hash."""
        try:
            key_path = os.path.join(mount_path, HardwareSecurity.KEY_FILENAME)
            if not os.path.exists(key_path):
                return False
            with open(key_path, "r") as f:
                content = f.read().strip()
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            return secrets.compare_digest(file_hash, HardwareSecurity.EXPECTED_HASH)
        except Exception:
            return False

    @staticmethod
    def generate_key_file(path="E:/"):
        """Generates a new hardware key file and prints its hash."""
        token = secrets.token_hex(32)
        key_path = os.path.join(path, HardwareSecurity.KEY_FILENAME)
        with open(key_path, "w") as f:
            f.write(token)
        hash_val = hashlib.sha256(token.encode()).hexdigest()
        print("[HardwareSecurity] Key generated successfully.")
        print("Store this hash securely:", hash_val)
