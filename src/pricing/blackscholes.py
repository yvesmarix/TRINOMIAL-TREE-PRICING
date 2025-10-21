from __future__ import annotations
import datetime as dt
from typing import Dict, Tuple, Literal, Optional

import numpy as np
from scipy.stats import norm


class BlackScholesPricer:
    """
    Pricer Black–Scholes pour options européennes (call/put),
    avec possibilité d’un dividende discret unique.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        dividend: float = 0.0,
        dividend_date: Optional[dt.datetime] = None,
    ) -> None:
        """
        Initialise le modèle Black–Scholes.

        Parameters
        ----------
        S : float
            Prix spot du sous-jacent.
        K : float
            Strike de l’option.
        T : float
            Maturité (en années).
        r : float
            Taux sans risque.
        sigma : float
            Volatilité annuelle.
        option_type : {'call', 'put'}
            Type d’option.
        dividend : float, optional
            Montant du dividende (par défaut 0).
        dividend_date : datetime, optional
            Date du dividende si applicable.
        """
        self.S: float = float(S)
        self.K: float = float(K)
        self.T: float = float(T)
        self.r: float = float(r)
        self.sigma: float = float(sigma)
        self.option_type: Literal["call", "put"] = option_type.lower()  # type: ignore
        self.dividend: float = float(dividend)
        self.dividend_date: Optional[dt.datetime] = dividend_date
        self._N, self._n = norm.cdf, norm.pdf

    # ------------------- utilitaires -------------------
    def _spot_adjusted(self) -> float:
        """
        Calcule le spot ajusté en tenant compte d’un dividende discret.
        """
        S = self.S
        if self.dividend and self.dividend_date:
            tD = max((self.dividend_date - dt.datetime.today()).days / 365.0, 0.0)
            S -= self.dividend * np.exp(-self.r * tD)
        return S

    def _d1d2(self, S_adj: float) -> Tuple[float, float]:
        """
        Renvoie les paramètres d1 et d2 du modèle Black–Scholes.
        """
        T = max(self.T, 1e-10)
        sig = max(self.sigma, 1e-12)
        d1 = (np.log(S_adj / self.K) + (self.r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        return d1, d2

    # ------------------- pricing -------------------
    def price(self) -> float:
        """
        Calcule le prix théorique (call ou put) selon Black–Scholes.
        """
        S_adj = self._spot_adjusted()
        d1, d2 = self._d1d2(S_adj)
        if self.option_type == "call":
            return S_adj * self._N(d1) - self.K * np.exp(-self.r * self.T) * self._N(d2)
        elif self.option_type == "put":
            return self.K * np.exp(-self.r * self.T) * self._N(-d2) - S_adj * self._N(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # ------------------- grecs -------------------
    def greeks(self) -> Dict[str, float]:
        """
        Calcule les grecs analytiques : delta, gamma, theta, vega, rho.
        Theta est ramené à un jour, vega et rho par 1% de variation.
        """
        S_adj = self._spot_adjusted()
        d1, d2 = self._d1d2(S_adj)
        T, sig = self.T, self.sigma

        if self.option_type == "call":
            delta = self._N(d1) * (S_adj / self.S)
            theta = -S_adj * self._n(d1) * sig / (2 * np.sqrt(T)) - self.r * self.K * np.exp(-self.r * T) * self._N(d2)
            rho = self.K * T * np.exp(-self.r * T) * self._N(d2)
        else:
            delta = -self._N(-d1) * (S_adj / self.S)
            theta = -S_adj * self._n(d1) * sig / (2 * np.sqrt(T)) + self.r * self.K * np.exp(-self.r * T) * self._N(-d2)
            rho = -self.K * T * np.exp(-self.r * T) * self._N(-d2)

        gamma = self._n(d1) / (S_adj * sig * np.sqrt(T)) * (S_adj / self.S)
        vega = S_adj * np.sqrt(T) * self._n(d1)
        vanna = self._n(d1) * np.sqrt(T) * (1 - d1 / (sig * np.sqrt(T)))

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta) / 365.0,  # par jour
            "vega": float(vega) / 100.0,    # par 1% de vol
            "rho": float(rho) / 100.0,      # par 1% de taux
            "vanna": float(vanna) / 100.0,  # par 1% de vol
        }


    # ------------------- mise à jour -------------------
    def update(self, **kwargs) -> "BlackScholesPricer":
        """
        Met à jour un ou plusieurs paramètres du pricer.
        Ex : bs.update(K=105, sigma=0.25, option_type='put')
        """
        allowed = {"S", "K", "T", "r", "sigma", "option_type", "dividend", "dividend_date"}
        for k, v in kwargs.items():
            if k not in allowed:
                raise AttributeError(f"Unknown parameter: {k}")
            if k == "option_type":
                self.option_type = str(v).lower()  # type: ignore
            else:
                setattr(self, k, v)
        return self
