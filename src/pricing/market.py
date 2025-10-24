import datetime as dt


class Market:
    """Conditions de marché : spot, taux, volatilité et dividende éventuel."""

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        dividend: float | None = None,
        dividend_date: dt.date | None = None,
    ) -> None:
        """Initialise le marché avec les paramètres principaux."""
        self.S0 = float(S0)
        self.r = float(r)
        self.sigma = float(sigma)
        self.dividend = float(dividend) if dividend is not None else 0.0
        self.dividend_date = dividend_date
