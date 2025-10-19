import numpy as np
from scipy.stats import norm
import datetime as dt

class BlackScholesPricer:
    def __init__(self,
                 S: float, K: float, T: float, r: float, sigma: float,
                 option_type: str, dividend: float = 0.0,
                 dividend_date: dt.datetime | None = None):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        self.dividend = float(dividend)
        self.dividend_date = dividend_date
        self._N = norm.cdf
        self._n = norm.pdf

    # ------- utils -------
    def _spot_adjusted(self) -> float:
        S = self.S
        if self.dividend and self.dividend_date is not None:
            tD = max(((self.dividend_date - dt.datetime.today()).days / 365.0), 0.0)
            S = S - self.dividend * np.exp(-self.r * tD)
        return S

    def _d1d2(self, S_adj: float) -> tuple[float, float]:
        T = max(self.T, 1e-10)
        sig = max(self.sigma, 1e-12)
        d1 = (np.log(S_adj / self.K) + (self.r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        return d1, d2

    # ------- prix -------
    def price(self) -> float:
        S_adj = self._spot_adjusted()
        d1, d2 = self._d1d2(S_adj)
        if self.option_type == "call":
            return S_adj * self._N(d1) - self.K * np.exp(-self.r * self.T) * self._N(d2)
        elif self.option_type == "put":
            return self.K * np.exp(-self.r * self.T) * self._N(-d2) - S_adj * self._N(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # ------- grecs analytiques -------
    def greeks(self) -> dict:
        S_adj = self._spot_adjusted()
        d1, d2 = self._d1d2(S_adj)
        T, sig = self.T, self.sigma

        if self.option_type == "call":
            delta = self._N(d1) * (S_adj / self.S)  # chaîne via S_adj(S)
            theta = -S_adj * self._n(d1) * sig / (2 * np.sqrt(T)) - self.r * self.K * np.exp(-self.r * T) * self._N(d2)
            rho   =  self.K * T * np.exp(-self.r * T) * self._N(d2)
        else:
            delta = -self._N(-d1) * (S_adj / self.S)
            theta = -S_adj * self._n(d1) * sig / (2 * np.sqrt(T)) + self.r * self.K * np.exp(-self.r * T) * self._N(-d2)
            rho   = -self.K * T * np.exp(-self.r * T) * self._N(-d2)

        gamma = self._n(d1) / (S_adj * sig * np.sqrt(T)) * (S_adj / self.S)
        vega  = S_adj * np.sqrt(T) * self._n(d1)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta) / 365.0,  # par jour
            "vega":  float(vega)  / 100.0,  # par 1% de vol
            "rho":   float(rho)   / 100.0,  # par 1% de taux
        }
    
    def update(self, **kwargs):
        """Met à jour un ou plusieurs paramètres : bs.update(K=105, sigma=0.22, ...)"""
        allowed = {"S","K","T","r","sigma","option_type","dividend","dividend_date"}
        for k, v in kwargs.items():
            if k not in allowed:
                raise AttributeError(f"Unknown parameter: {k}")
            if k == "option_type":
                self.option_type = str(v).lower()
            else:
                setattr(self, k, v)
        return self
