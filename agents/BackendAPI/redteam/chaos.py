import random
import time
import numpy as np

class ChaosMonkey:
    """Chaos Engineering – test odporności systemu"""

    @staticmethod
    def inject_flash_crash(df):
        """Symuluje nagły spadek ceny o 90% na ostatniej świecy"""
        df_corrupted = df.copy()
        if not df_corrupted.empty:
            last_idx = df_corrupted.index[-1]
            df_corrupted.at[last_idx, "close"] = df_corrupted.at[last_idx, "close"] * 0.10
        return df_corrupted

    @staticmethod
    def inject_latency():
        """Symuluje opóźnienie sieciowe (lag)"""
        lag = random.uniform(1.5, 5.0)
        time.sleep(lag)
        return lag

    @staticmethod
    def inject_bad_data(df):
        """Wstrzykuje uszkodzone dane (NaN)"""
        df_corrupted = df.copy()
        if not df_corrupted.empty:
            last_idx = df_corrupted.index[-1]
            df_corrupted.at[last_idx, "close"] = np.nan
        return df_corrupted
