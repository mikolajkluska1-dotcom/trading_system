class CapitalGuard:
    """
    V68: Dynamiczna Agresja oparta na MQS.
    Wersja HARDENED: Odporna na błędy matematyczne i błędne dane API.
    """

    @staticmethod
    def calculate_position_size(balance, price, mqs_score):
        # 1. Walidacja Krytyczna Danych Wejściowych (Anti-Crash)
        if balance <= 0:
            return 0.0, 0.0
        if price <= 0: # Ochrona przed dzieleniem przez zero
            return 0.0, 0.0

        # Clamp mqs_score do zakresu 0-100 (na wszelki wypadek)
        mqs_score = max(0, min(100, mqs_score))

        # Baza: Maksymalnie 15% kapitału na pozycję w idealnych warunkach
        MAX_RISK = 0.15

        # Funkcja agresji: Kwadratowa (karze niskie wyniki, nagradza wysokie)
        # Przy MQS 50 -> aggression = 0.25 (25% mocy)
        aggression_factor = (mqs_score / 100.0) ** 2

        usd_size = balance * MAX_RISK * aggression_factor

        # Hard Limit: Nie więcej niż 98% salda (zostawiamy na fees)
        if usd_size > balance * 0.98:
            usd_size = balance * 0.98

        # Min. wielkość zlecenia (np. $11 dla Binance)
        if usd_size < 11.0:
            usd_size = 0.0

        # Obliczenie ilości (zabezpieczone, bo price > 0 sprawdzone wyżej)
        qty = usd_size / price

        # Zaokrąglenie finansowe (2 miejsca dla USD, 6 dla krypto)
        return round(usd_size, 2), round(qty, 6)
