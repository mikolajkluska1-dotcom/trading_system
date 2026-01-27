import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json
import os

class GoogleReporter:
    def __init__(self):
        self.scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        self.creds_file = os.path.join("assets", "google_credentials.json")
        self.client = None
        self.sheet = None
        
        if os.path.exists(self.creds_file):
            try:
                self.creds = ServiceAccountCredentials.from_json_keyfile_name(self.creds_file, self.scope)
                self.client = gspread.authorize(self.creds)
                self.sheet = self.client.open("Redline_Trading_Logs").sheet1
            except Exception as e:
                print(f"⚠️ Google Sheets Auth Failed: {e}")

    def log_trade(self, symbol, action, price, amount, profit_usd, ai_score, confidence, reasons):
        """
        Logs trade data matching the user's exact column requirements.
        """
        if not self.sheet: return False

        try:
            # 1. Define Exact Headers
            headers = [
                "Data", "Symbol", "Akcja", "Cena", "Ilość", "Koszt", 
                "Profit", "AI Score", "Pewność (Conf)", "Powód Decyzji", "Zysk ($)"
            ]
            
            # Check if we need to initialize the sheet
            current_headers = self.sheet.row_values(1)
            if not current_headers or current_headers[0] != "Data":
                self.sheet.clear() # Reset if schema doesn't match
                self.sheet.append_row(headers)

            # 2. Calculate Derived Metrics
            cost = float(price) * float(amount)
            
            # Calculate ROI % for "Profit" column
            # Avoid division by zero
            roi_pct = (float(profit_usd) / cost * 100) if cost > 0 else 0.0

            # 3. Prepare Row Data
            row = [
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), # Data
                str(symbol),              # Symbol
                str(action),              # Akcja
                float(price),             # Cena
                float(amount),            # Ilość
                round(cost, 2),           # Koszt
                f"{roi_pct:.2f}%",        # Profit (ROI)
                int(ai_score),            # AI Score
                float(confidence),        # Pewność (Conf)
                str(reasons),             # Powód Decyzji
                round(float(profit_usd), 2) # Zysk ($)
            ]
            
            # 4. Append
            self.sheet.append_row(row)
            return True
            
        except Exception as e:
            print(f"❌ Sheets Log Error: {e}")
            return False
