import numpy as np

def compute_forward(S: float, r: float, dt: float) -> float:
    return np.exp(r * dt) * S

def compute_variance(S: float, r: float, dt: float, sigma: float) -> float:
        return (
        S**2
        * np.exp(2 * r * dt)
        * (np.exp(sigma**2 * dt) - 1)
        )

def compute_p_down(esperance: float, forward: float, variance: float, alpha: float) -> float:
        return(
            esperance ** (-2) * (variance + forward**2)
            - 1
            - (alpha + 1) * (esperance ** (-1) * forward - 1)
        ) / ((1 - alpha) * (alpha ** (-2) - 1))

def compute_p_up(p_down: float, alpha: float, esperance: float = None
                , forward: float = None, dividend: bool = False) -> float:
    if not dividend:
        return p_down / alpha
    else:
        return ((alpha - 1)**(-1)) * ((esperance ** (-1)) * forward - 1 -
                ((alpha ** (-1)) - 1) * p_down)

def compute_p_mid(p_down: float, p_up: float) -> float:
     return 1 - p_down - p_up

def compute_probabilities(esperance: float, forward: float
                          , variance: float, alpha: float
                          , dividend: bool = False):
    p_down = compute_p_down(esperance, forward, variance, alpha)
    p_up = compute_p_up(p_down, alpha, esperance, forward, dividend)
    p_mid = compute_p_mid(p_down, p_up)

    return p_down, p_up, p_mid