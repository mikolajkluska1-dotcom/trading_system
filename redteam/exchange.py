import random

class ExchangeSimulator:
    """
    Symulator wrogiej giełdy (Adversarial Broker).
    Testuje odporność systemu na: Slippage, Partial Fills, Rejections.
    """

    @staticmethod
    def execute_order(side, price, qty):
        """
        Symuluje egzekucję zlecenia w nieidealnych warunkach.
        Zwraca: {status, filled_qty, avg_price, reason}
        """
        # Rzut kostką losu (Entropy)
        roll = random.random()

        # SCENARIUSZ 1: ORDER REJECTED (10% szans)
        # Np. brak płynności, błąd API, overload silnika giełdy.
        if roll < 0.10:
            return {
                "status": "REJECTED",
                "filled_qty": 0.0,
                "avg_price": 0.0,
                "reason": "EXCHANGE_ENGINE_OVERLOAD"
            }

        # SCENARIUSZ 2: PARTIAL FILL + SLIPPAGE (20% szans)
        # Zlecenie wypełnione tylko częściowo i po gorszej cenie.
        if roll < 0.30:
            # Wypełniamy od 10% do 50% zlecenia
            filled = qty * random.uniform(0.1, 0.5)
            
            # Poślizg cenowy 0.2% - 1.0% (przeciwko nam)
            slippage = random.uniform(0.002, 0.010)
            if side == "BUY":
                exec_price = price * (1 + slippage)
            else:
                exec_price = price * (1 - slippage)

            return {
                "status": "PARTIAL_FILL",
                "filled_qty": round(filled, 6),
                "avg_price": round(exec_price, 2),
                "reason": "INSUFFICIENT_LIQUIDITY"
            }

        # SCENARIUSZ 3: FULL FILL + STANDARD SLIPPAGE (70% szans)
        # Nawet przy pełnym wypełnieniu cena rzadko jest idealna.
        slippage = random.uniform(0.0005, 0.002) # 0.05% - 0.2%
        if side == "BUY":
            exec_price = price * (1 + slippage)
        else:
            exec_price = price * (1 - slippage)

        return {
            "status": "FILLED",
            "filled_qty": qty,
            "avg_price": round(exec_price, 2),
            "reason": "OK"
        }