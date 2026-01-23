import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# Logger - żebyś widział w konsoli co myśli mózg
logger = logging.getLogger("DEEP_BRAIN_V2")
logging.basicConfig(level=logging.INFO)

# --- KONFIGURACJA V2 (Musi pasować do treningu) ---
MODEL_PATH = "R:/REDLINE_SYSTEM/ai_models/btc_lstm_v2.pth"
SEQ_LENGTH = 60     # Patrzymy 60 minut wstecz
HIDDEN_SIZE = 100   # ZWIĘKSZONY ROZMIAR
NUM_LAYERS = 2      # ZWIĘKSZONA GŁĘBOKOŚĆ

# --- DEFINICJA SIECI (Bliźniak tej z treningu) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(CryptoLSTM, self).__init__()
        # Dokładnie taka sama struktura jak przy uczeniu
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Bierzemy tylko ostatni krok czasowy
        out = self.fc(out[:, -1, :])
        return out

class DeepBrain:
    """
    Sterownik V2 - Obsługuje model 'Massive Brain' (900k świeczek).
    """
    def __init__(self):
        self.device = torch.device("cpu") # Do odczytu CPU jest wystarczające
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ [DeepBrain] Brak modelu V2 w: {MODEL_PATH}")
            return

        try:
            self.model = CryptoLSTM(hidden_size=HIDDEN_SIZE)
            # Ładowanie wag (map_location=cpu chroni przed błędami, gdybyś trenował na GPU)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval() # Tryb ewaluacji (wyłącza losowość dropoutu)
            logger.info(f"✅ [DeepBrain] MÓZG V2 ZAŁADOWANY (100 Neuronów / 2 Warstwy)")
        except Exception as e:
            logger.error(f"❌ [DeepBrain] Błąd ładowania V2: {e}")
            self.model = None

    def predict(self, df):
        """
        Zwraca: (predicted_price, confidence, signal)
        """
        if self.model is None:
            return 0, 0, "NEUTRAL"

        if len(df) < SEQ_LENGTH:
            return 0, 0, "NEUTRAL"

        try:
            # 1. Przygotowanie danych (Ostatnie 60 świeczek 'close')
            data = df['close'].tail(SEQ_LENGTH).values.reshape(-1, 1)
            current_price = data[-1][0]

            # 2. Skalowanie (Fitujemy na bieżącym oknie - lokalny kontekst jest najważniejszy)
            self.scaler.fit(data)
            scaled_data = self.scaler.transform(data)

            # 3. Konwersja na Tensor
            X_input = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)

            # 4. Predykcja
            with torch.no_grad():
                prediction_scaled = self.model(X_input)
                predicted_price = self.scaler.inverse_transform(prediction_scaled.numpy())[0][0]

            # 5. Logika Decyzyjna V2
            diff_percent = ((predicted_price - current_price) / current_price) * 100
            
            signal = "NEUTRAL"
            confidence = 0.5
            threshold = 0.10 # Próg reakcji 0.1%

            if diff_percent > threshold:
                signal = "BUY"
                confidence = min(0.6 + (diff_percent * 2.5), 0.99)
            elif diff_percent < -threshold:
                signal = "SELL"
                confidence = min(0.6 + (abs(diff_percent) * 2.5), 0.99)

            logger.info(f"🧠 AI V2: {current_price:.2f}$ -> {predicted_price:.2f}$ ({diff_percent:+.2f}%) [{signal}] Conf: {confidence:.2f}")
            
            return predicted_price, confidence, signal

        except Exception as e:
            logger.error(f"❌ [DeepBrain] Błąd predykcji: {e}")
            return 0, 0, "ERROR"