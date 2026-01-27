import random
import requests

class OsintOps:
    """Moduł OSINT / Recon & Stealth"""

    @staticmethod
    def generate_dorks(target):
        """Generuje zapytania Google Dorking dla celu"""
        base = "https://www.google.com/search?q="
        queries = [
            ("CONFIDENTIAL PDF", f'site:*.{target}.com filetype:pdf "confidential"'),
            ("EXPOSED DB", f'site:*.{target}.com ext:sql | ext:dbf | ext:mdb'),
            ("LOGIN PORTALS", f'site:*.{target}.com inurl:login'),
            ("GITHUB SECRETS", f'site:github.com "{target}" "API_KEY"')
        ]

        return [(name, base + requests.utils.quote(q)) for name, q in queries]

    @staticmethod
    def stealth_headers():
        """Generuje fałszywe nagłówki przeglądarki (User-Agent)"""
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        ]

        return {
            "User-Agent": random.choice(agents)
        }
