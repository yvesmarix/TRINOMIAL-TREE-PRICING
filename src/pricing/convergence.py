from __future__ import annotations

import datetime as dt
import time
from typing import TYPE_CHECKING, Callable, Iterable, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from pricing import BlackScholesPricer, TrinomialTree

if TYPE_CHECKING:  # évite les imports circulaires au runtime
    from pricing.market import Market
    from pricing.option import Option


def _setup_bs(market: "Market", option: "Option") -> Tuple[dt.date, float, BlackScholesPricer]:
    """Prépare date de pricing, T (années) et pricer Black–Scholes."""
    pricing_date = dt.date.today()
    T = (option.maturity - pricing_date).days / 365
    bs = BlackScholesPricer(
        S=market.S0, K=option.K, T=T, r=market.r, sigma=market.sigma,
        option_type=option.option_type,
        dividend=getattr(market, "dividend", 0.0),
        dividend_date=getattr(market, "dividend_date", None),
    )
    return pricing_date, T, bs


def _make_tree_factory(market: "Market", pricing_date: dt.date,
                       pruning: bool, epsilon: float) -> Callable[[int], TrinomialTree]:
    """Fabrique d’arbres pour éviter de répéter le constructeur."""
    def _factory(n_steps: int) -> TrinomialTree:
        return TrinomialTree(market, N=n_steps, pruning=pruning,
                             epsilon=epsilon, pricing_date=pricing_date)
    return _factory


def _plot_base(title: str, xlabel: str, ylabel: str, logy: bool = False):
    """Crée fig/ax avec mise en forme de base."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if logy: ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def _plot_strike_curve(k_vals: Sequence[float], bs_vals: Sequence[float],
                       tree_vals: Sequence[float], n_steps: int) -> None:
    """Trace BS vs Tree en fonction du strike."""
    fig, ax = _plot_base(f"Prix vs Strike (N={n_steps})", "Strike K", "Prix")
    ax.plot(k_vals, bs_vals, label="Black–Scholes", lw=2, color="steelblue")
    ax.scatter(k_vals, tree_vals, label="Trinomial Tree", s=25, color="darkorange")
    ax.legend(loc="upper left"); fig.tight_layout(); plt.show()


def _plot_convergence_price(n_vals: Sequence[int], tree_px: Sequence[float],
                            bs_price: float) -> None:
    """Trace la convergence du prix en fonction de N."""
    fig, ax = _plot_base("Convergence du prix vs N", "N", "Prix")
    ax.plot(n_vals, tree_px, color="darkorange", label="Trinomial Tree")
    ax.axhline(bs_price, color="steelblue", ls="--", label="Black–Scholes")
    ax.legend(loc="upper left"); fig.tight_layout(); plt.show()


def _plot_convergence_error(n_vals: Sequence[int], abs_err: Sequence[float]) -> None:
    """Trace l'erreur absolue en échelle log."""
    fig, ax = _plot_base("Erreur absolue (échelle log)", "N", "|Erreur|", logy=True)
    ax.plot(n_vals, abs_err, color="crimson"); fig.tight_layout(); plt.show()


def _build_option_like(option: "Option", K: float) -> "Option":
    """Recrée une option identique mais avec un K différent."""
    return option.__class__(
        K=K, option_type=option.option_type,
        maturity=option.maturity, option_class=option.option_class,
    )


def _compute_strike_curves(option: "Option", strikes: Iterable[float],
                           bs: BlackScholesPricer, tree: TrinomialTree):
    """Retourne (K, BS(K), Tree(K)) pour une liste de strikes."""
    k_vals: List[float] = []; bs_vals: List[float] = []; tree_vals: List[float] = []
    for k in strikes:
        opt_k = _build_option_like(option, k)
        bs.update(K=k); k_vals.append(k); bs_vals.append(bs.price())
        tree_vals.append(tree.price(opt_k, build_tree=True))
    return k_vals, bs_vals, tree_vals


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def bs_convergence_by_strike(
    market: "Market",
    option: "Option",
    strikes: Iterable[float],
    n_steps: int = 200,
    pruning: bool = True,
    epsilon: float = 1e-7,
) -> None:
    """Compare BS et Trinomial en fonction du strike."""
    pricing_date, _, bs = _setup_bs(market, option)
    tree = _make_tree_factory(market, pricing_date, pruning, epsilon)(n_steps)
    k_vals, bs_vals, tree_vals = _compute_strike_curves(option, strikes, bs, tree)
    _plot_strike_curve(k_vals, bs_vals, tree_vals, n_steps)


def bs_convergence_by_step(
    market: "Market",
    option: "Option",
    max_n: int = 400,
    step: int = 25,
    pruning: bool = True,
    epsilon: float = 1e-7,
) -> None:
    """Étudie la convergence du trinomial vers BS en fonction de N."""
    pricing_date, _, bs = _setup_bs(market, option); bs_price = bs.price()
    n_vals = np.arange(step, max_n + 1, step, dtype=int)
    make_tree = _make_tree_factory(market, pricing_date, pruning, epsilon)
    tree_prices = [make_tree(int(n)).price(option, build_tree=True) for n in n_vals]
    abs_errors = np.abs(np.array(tree_prices) - bs_price)
    _plot_convergence_price(n_vals, tree_prices, bs_price)
    _plot_convergence_error(n_vals, abs_errors)


def plot_runtime_vs_steps(
    market: "Market",
    option: "Option",
    N_values: Sequence[int],
    method: str = "backward",
    build_tree: bool = True,
    compute_greeks: bool = False,
) -> None:
    """Affiche le temps de price() en fonction de N (log-log)."""
    times: List[float] = []
    for N in N_values:
        tree = TrinomialTree(market, N)
        start = time.perf_counter()
        tree.price(option, method=method, build_tree=build_tree, compute_greeks=compute_greeks)
        times.append(time.perf_counter() - start)
    plt.figure(figsize=(8, 5))
    plt.loglog(N_values, times, marker="o")
    plt.xlabel("Nombre de pas N (log)"); plt.ylabel("Temps (s, log)")
    plt.title("Temps d'exécution vs Nombre de pas (log-log)")
    plt.grid(True, which="both", ls="--", alpha=0.5); plt.tight_layout(); plt.show()
