import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# Ustawienie loggera, żebyś widział co robi mózg w konsoli
logger = logging.getLogger("DEEP_BRAIN")
logging.basicConfig(level=logging.INFO)

# --- KONFIGURACJA MODELU ---
# To musi celować w Twój dysk R:
MODEL_PATH = "R:/REDLINE_SYSTEM/ai_models/btc_lstm_v1.pth"
SEQ_LENGTH = 60    # Pamięć bota (60 świeczek wstecz)
HIDDEN_SIZE = 50   # Ilość neuronów

# --- DEFINICJA SIECI (Musi być BLIŹNIAKIEM tej z treningu) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class DeepBrain:
    """
    To jest 'Sterownik' do Twojego modelu na dysku R.
    Łączy plik .pth z resztą systemu.
    """
    def __init__(self):
        self.device = torch.device("cpu") # Do odczytu wystarczy CPU (jest stabilniejsze)
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._load_model()

    def _load_model(self):
        """Próbuje wczytać mózg z dysku R"""
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ [DeepBrain] Nie znaleziono pliku modelu: {MODEL_PATH}")
            return

        try:
            self.model = CryptoLSTM(hidden_size=HIDDEN_SIZE)
            # Ładowanie wag
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval() # Tryb 'Egzaminu' (nie ucz się teraz, tylko odpowiadaj)
            logger.info(f"✅ [DeepBrain] Mózg załadowany pomyślnie z: {MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ [DeepBrain] Krytyczny błąd ładowania modelu: {e}")
            self.model = None

    def predict(self, df):
        """
        Główna funkcja, o którą pyta Bot.
        Dostaje tabelkę z cenami -> Zwraca przewidywanie.
        """
        # Zabezpieczenie: Jeśli mózg nie działa, zwróć 'neutral'
        if self.model is None:
            return 0, 0, "NEUTRAL"

        # Zabezpieczenie: Czy mamy wystarczająco dużo danych?
        if len(df) < SEQ_LENGTH:
            # logger.warning(f"⚠️ [DeepBrain] Za mało danych: {len(df)} (Wymagane: {SEQ_LENGTH})")
            return 0, 0, "NEUTRAL"

        try:
            # 1. Wycinamy ostatnie 60 świeczek ceny zamknięcia (close)
            data = df['close'].tail(SEQ_LENGTH).values.reshape(-1, 1)
            current_price = data[-1][0]

            # 2. Skalowanie (zamiana ceny $90000 na liczbę 0.0-1.0)
            self.scaler.fit(data) 
            scaled_data = self.scaler.transform(data)

            # 3. Pakowanie w Tensor (format dla PyTorch)
            X_input = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)

            # 4. Magia AI (Przewidywanie)
            with torch.no_grad():
                prediction_scaled = self.model(X_input)
                # Odwracamy skalowanie (zamiana wyniku 0.5 na dolary)
                predicted_price = self.scaler.inverse_transform(prediction_scaled.numpy())[0][0]

            # 5. Logika Decyzyjna (Czy wzrost jest wystarczająco duży?)
            diff_percent = ((predicted_price - current_price) / current_price) * 100
            
            signal = "NEUTRAL"
            confidence = 0.5 # Bazowa pewność siebie bota

            # Jeśli przewiduje ruch większy niż 0.1% w górę/dół
            if diff_percent > 0.1:
                signal = "BUY"
                # Pewność rośnie wraz z siłą przewidywanego ruchu
                confidence = min(0.5 + (diff_percent * 2), 0.95)
            elif diff_percent < -0.1:
                signal = "SELL"
                confidence = min(0.5 + (abs(diff_percent) * 2), 0.95)

            # Wypisz w logach co myśli bot
            logger.info(f"🧠 AI: Cena: {current_price:.2f} -> Przewidywana: {predicted_price:.2f} (Zmiana: {diff_percent:+.2f}%) -> Decyzja: {signal}")
            
            return predicted_price, confidence, signal

        except Exception as e:
            logger.error(f"❌ [DeepBrain] Błąd podczas analizy: {e}")
            return 0, 0, "ERROR"