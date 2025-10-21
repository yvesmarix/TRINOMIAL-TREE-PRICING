import datetime as dt


class Option:
    """Représente une option (Call/Put, Européenne/Américaine)."""

    def __init__(self, K: float, maturity: dt.datetime, option_type: str, option_class: str) -> None:
        """Initialise l’option avec strike, maturité, type et classe."""
        self.K = float(K)
        self.maturity = maturity
        self.option_type = self._check_option_type(option_type)
        self.option_class = self._check_option_class(option_class)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_option_type(option_type: str) -> str:
        """Vérifie que le type est 'call' ou 'put'."""
        t = option_type.lower()
        if t not in ("call", "put"):
            raise ValueError("option_type doit être 'call' ou 'put'.")
        return t

    @staticmethod
    def _check_option_class(option_class: str) -> str:
        """Vérifie que la classe est 'european' ou 'american'."""
        c = option_class.lower()
        if c not in ("european", "american"):
            raise ValueError("option_class doit être 'european' ou 'american'.")
        return c

    # ------------------------------------------------------------------ #
    # Payoff
    # ------------------------------------------------------------------ #
    def payoff(self, S: float) -> float:
        """Renvoie le payoff de l’option pour un prix spot donné."""
        if self.option_type == "call":
            return max(S - self.K, 0.0)
        if self.option_type == "put":
            return max(self.K - S, 0.0)
        raise ValueError("Type d’option invalide : 'call' ou 'put' attendu.")
