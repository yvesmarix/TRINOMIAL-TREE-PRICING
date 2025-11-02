import datetime as dt
import time
import numpy as np
import matplotlib.pyplot as plt

from pricing import BlackScholesPricer, TrinomialTree


def _setup_bs(market, option):
    """
    Prépare la date de pricing, l'échéance T (en années) et le pricer Black–Scholes.
    """
    pricing_date = dt.date.today()
    T = (option.maturity - pricing_date).days / 365
    bs = BlackScholesPricer(
        S=market.S0, K=option.K, T=T, r=market.r, sigma=market.sigma,
        option_type=option.option_type, dividend=getattr(market, "dividend", 0.0),
        dividend_date=getattr(market, "dividend_date", None),
    )
    return pricing_date, T, bs


def _make_tree_factory(market, pricing_date, pruning: bool, epsilon: float):
    """
    Renvoie une petite fabrique d'arbres pour éviter de répéter le constructeur.
    """
    def _factory(n_steps: int) -> TrinomialTree:
        return TrinomialTree(
            market, N=n_steps, pruning=pruning, epsilon=epsilon, pricingDate=pricing_date,
        )
    return _factory


def _plot_strike_curve(k_vals, bs_vals, tree_vals, n_steps: int):
    """Trace BS vs Tree en fonction du strike."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_vals, bs_vals, label="Black–Scholes", lw=2, color="steelblue")
    ax.scatter(k_vals, tree_vals, label="Trinomial Tree", s=25, color="darkorange")
    ax.set(title=f"Prix vs Strike (N={n_steps})", xlabel="Strike K", ylabel="Prix")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def _plot_convergence_price(n_vals, tree_prices, bs_price):
    """Trace la convergence du prix en fonction de N."""
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(n_vals, tree_prices, color="darkorange", label="Trinomial Tree")
    ax1.axhline(bs_price, color="steelblue", ls="--", label="Black–Scholes")
    ax1.set(title="Convergence du prix vs N", xlabel="N", ylabel="Prix")
    ax1.legend(loc="upper left")
    fig1.tight_layout()
    plt.show()


def _plot_convergence_error(n_vals, abs_errors):
    """Trace l'erreur absolue en échelle log."""
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(n_vals, abs_errors, color="crimson")
    ax2.set_yscale("log")
    ax2.set(title="Erreur absolue (échelle log)", xlabel="N", ylabel="|Erreur|")
    fig2.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #
def bs_convergence_by_strike(
    market, option, strikes, n_steps=200, pruning=True, epsilon=1e-7
):
    """
    Compare les prix Black–Scholes et Trinomial en fonction du strike.
    """
    pricing_date, _, bs = _setup_bs(market, option)
    make_tree = _make_tree_factory(market, pricing_date, pruning, epsilon)
    tree = make_tree(n_steps)

    k_vals, bs_vals, tree_vals = [], [], []
    for k in strikes:
        opt_k = option.__class__(
            K=k, option_type=option.option_type,
            maturity=option.maturity, option_class=option.option_class,
        )
        bs.update(K=k); k_vals.append(k); bs_vals.append(bs.price())
        tree_vals.append(tree.price(opt_k, build_tree=True))

    _plot_strike_curve(k_vals, bs_vals, tree_vals, n_steps)


def bs_convergence_by_step(
    market, option, max_n=400, step=25, pruning=True, epsilon=1e-7
):
    """
    Étudie la convergence du trinomial vers Black–Scholes en fonction de N.
    """
    pricing_date, _, bs = _setup_bs(market, option)
    bs_price = bs.price()

    n_vals = np.arange(step, max_n + 1, step, dtype=int)
    make_tree = _make_tree_factory(market, pricing_date, pruning, epsilon)

    tree_prices = [
        make_tree(int(n)).price(option, build_tree=True)
        for n in n_vals
    ]
    abs_errors = np.abs(np.array(tree_prices) - bs_price)

    _plot_convergence_price(n_vals, tree_prices, bs_price)
    _plot_convergence_error(n_vals, abs_errors)


def plot_runtime_vs_steps(
    market, option, N_values, method="backward", build_tree=True, compute_greeks=False
):
    """
    Affiche le temps d'exécution de price() en fonction du nombre de pas N (échelle log-log).
    """
    times = []
    for N in N_values:
        tree = TrinomialTree(market, N)
        start = time.perf_counter()
        tree.price(option, method=method, build_tree=build_tree, compute_greeks=compute_greeks)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    plt.figure(figsize=(8, 5))
    plt.loglog(N_values, times, marker='o')
    plt.xlabel("Nombre de pas N (log)")
    plt.ylabel("Temps d'exécution (s, log)")
    plt.title("Temps d'exécution vs Nombre de pas (log-log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
