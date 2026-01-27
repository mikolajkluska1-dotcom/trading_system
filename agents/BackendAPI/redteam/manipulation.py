import numpy as np
import pandas as pd

class AdversarialMarket:
    """
    Generuje dane rynkowe typu 'Bull Trap'.
    Celem jest oszukanie modelu ML, by dał sygnał BUY na szczycie.
    """

    @staticmethod
    def generate_false_trend(df):
        """
        Modyfikuje ostatnie 15 świec, tworząc idealny, sztuczny trend.
        """
        df_adv = df.copy()
        n = len(df_adv)

        # Potrzebujemy min. 20 świec do manipulacji
        if n < 20:
            return df_adv

        # 1. Mikro-wzrosty (Stable Grind Up)
        # To oszukuje algorytmy oparte na średnich kroczących.
        # Generujemy serię małych, pozytywnych zwrotów.
        fake_returns = np.random.normal(0.002, 0.0005, 15) # +0.2% avg

        # 2. Sztuczne podbijanie ceny (Painting the Tape)
        current_price = df_adv['close'].iloc[-16]
        for i in range(15):
            idx = n - 15 + i
            current_price = current_price * (1 + fake_returns[i])
            df_adv.at[df_adv.index[idx], 'close'] = current_price

            # High/Low też muszą wyglądać "bezpiecznie" (mała zmienność)
            df_adv.at[df_adv.index[idx], 'h'] = current_price * 1.001
            df_adv.at[df_adv.index[idx], 'l'] = current_price * 0.999

        # 3. Fałszowanie Wolumenu (Volume Masking)
        # Ustawiamy wolumen na średni poziom, żeby nie uruchomić alarmów "Low Volume".
        avg_vol = df_adv['v'].mean()
        df_adv.iloc[-15:, df_adv.columns.get_loc('v')] = avg_vol * np.random.uniform(0.9, 1.1, 15)

        # 4. Manipulacja RSI (Divergence Hiding)
        # Ręcznie ustawiamy RSI w strefie "Safe Bullish" (55-68), unikając Overbought (>70).
        # To najbardziej myli boty.
        fake_rsi = np.linspace(55, 68, 15)
        if 'rsi' in df_adv.columns:
            df_adv.iloc[-15:, df_adv.columns.get_loc('rsi')] = fake_rsi

        return df_adv
