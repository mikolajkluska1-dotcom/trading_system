import hashlib

class RedTeamOps:
    """Narzędzia audytowe (AUTHORIZED USE ONLY)"""

    @staticmethod
    def crack_hash(target_hash):
        """Prosty słownikowy łamacz haseł MD5"""
        dictionary = [
            "admin", "123456", "password", "redline",
            "root", "Cyber_Fortress_X1", "qwerty"
        ]

        for pwd in dictionary:
            if hashlib.md5(pwd.encode()).hexdigest() == target_hash:
                return pwd

        return None
