import pandas as pd
import numpy as np
from ml.brain import DeepBrain

def test_brain_logic():
    print("Testing DeepBrain V6 (Rich Context)...")

    # 1. Tworzymy sztuczne dane (Trend Wzrostowy)
    data = {
        'close': [100 + i + np.random.normal(0, 0.1) for i in range(100)],
        'high': [102 + i for i in range(100)],
        'low': [99 + i for i in range(100)],
        'volume': [1000 + i*10 for i in range(100)],
        'open': [100 + i for i in range(100)]
    }
    df = pd.DataFrame(data)

    # 2. Inicjalizacja Brain
    brain = DeepBrain(lookback=20)

    # 3. Predykcja
    try:
        price, conf, signal = brain.predict(df)
        print(f"Prediction Result: Price=${price:.2f}, Conf={conf:.2f}, Signal={signal}")

        if signal in ["BUY", "SELL", "HOLD"]:
            print("SUCCESS: Brain is operational with rich context features.")
        else:
            print(f"WARNING: Unexpected signal: {signal}")

    except Exception as e:
        print(f"FAILED: error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_brain_logic()
