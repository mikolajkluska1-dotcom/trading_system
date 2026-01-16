import requests
import datetime
import json

# CONFIGURATION
# User Instructions: Replace this URL with your deployed Google Apps Script Web App URL
SHEETS_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbykUc9iYhYcmjlokgMJVtfPMag45qd_KWGuWsIbbYfbN-EqLyHQkzzo6KjwHwB-O28GPw/exec"

class GoogleReporter:
    def __init__(self):
        self.webhook_url = SHEETS_WEBHOOK_URL

    def log_trade(self, symbol, action, price, amount, profit=0.0):
        """
        Sends trade details to Google Sheets via Webhook.
        """
        if "INSERT_YOUR" in self.webhook_url:
            print("⚠️ [G-SHEETS] Webhook URL not configured. Skipping log.")
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cost = float(price) * float(amount)
            
            # Payload matching Google Apps Script expectation
            payload = {
                "timestamp": timestamp,
                "symbol": str(symbol),
                "action": str(action),
                "price": float(price),
                "amount": float(amount),
                "cost": cost,
                "profit": float(profit)
            }
            
            # Send POST request
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            
            if response.status_code in [200, 201, 302]:
                print(f"✅ [G-SHEETS] Logged: {action} {symbol}")
            else:
                print(f"⚠️ [G-SHEETS] Failed to log. Status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ [G-SHEETS] Error logging trade: {e}")
