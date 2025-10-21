import numpy as np
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from trinomial_tree import TrinomialTree
    
def probas_valid(tree: Optional["TrinomialTree"]) -> bool:
    """Verifie que les probabilites sont valides et sommees a 1."""
    p_down, p_up, p_mid = tree.p_down, tree.p_up, tree.p_mid
    return (
        0 <= p_down <= 1 and 0 <= p_mid <= 1 and 0 <= p_up <= 1
        and abs(p_down + p_mid + p_up - 1.0) <= 1e-12
    )

def compute_forward(S: float, r: float, dt: float) -> float:
    """Calcule le forward attendu : S·e^(r·Δt)."""
    return S * np.exp(r * dt)


def compute_variance(S: float, r: float, dt: float, sigma: float) -> float:
    """Variance du sous-jacent sur Δt (processus log-normal)."""
    return S**2 * np.exp(2 * r * dt) * (np.exp(sigma**2 * dt) - 1)


def compute_p_down(esperance: float, forward: float, variance: float, alpha: float) -> float:
    """Probabilité p_down du trinomial à partir des moments cibles."""
    return (
        forward**(-2) * (variance + esperance**2)
        - 1
        - (alpha + 1) * (forward**(-1) * esperance - 1)
    ) / ((1 - alpha) * (alpha**(-2) - 1))


def compute_p_up(
    p_down: float,
    alpha: float,
    esperance: float | None = None,
    forward: float | None = None,
    dividend: bool = False,
) -> float:
    """Probabilité p_up (ajustée si dividende)."""
    if not dividend:
        return p_down / alpha
    x = esperance / forward
    return (x - 1) / (alpha - 1) + p_down / alpha


def compute_p_mid(p_down: float, p_up: float) -> float:
    """Probabilité p_mid = 1 - p_down - p_up."""
    return 1 - p_down - p_up


def compute_probabilities(
    esperance: float,
    forward: float,
    variance: float,
    alpha: float,
    dividend: bool = False,
) -> tuple[float, float, float]:
    """Retourne (p_down, p_up, p_mid) à partir des moments et α."""
    p_down = compute_p_down(esperance, forward, variance, alpha)
    p_up = compute_p_up(p_down, alpha, esperance, forward, dividend)
    p_mid = compute_p_mid(p_down, p_up)
    return p_down, p_up, p_mid


def iter_column(mid_node):
    """Itère sur tous les nœuds d’une colonne (d’abord vers le haut, puis vers le bas)."""
    n = mid_node
    while n:
        yield n
        n = n.up
    n = mid_node.down
    while n:
        yield n
        n = n.down
