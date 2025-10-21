import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from pricing import BlackScholesPricer, TrinomialTree


def bs_convergence_by_strike(
    market, option, strikes, n_steps=200, pruning=True, epsilon=1e-7
):
    """
    Compare les prix Black–Scholes et Trinomial en fonction du strike.
    """
    pricing_date = dt.datetime.today()
    T = (option.maturity - pricing_date).days / 365

    # Pricers (OOP)
    bs = BlackScholesPricer(
        S=market.S0, K=option.K, T=T, r=market.r, sigma=market.sigma,
        option_type=option.option_type,
        dividend=getattr(market, "dividend", 0.0),
        dividend_date=getattr(market, "dividend_date", None),
    )
    tree = TrinomialTree(
        market, N=n_steps, pruning=pruning,
        epsilon=epsilon, pricingDate=pricing_date,
    )

    k_vals, bs_vals, tree_vals = [], [], []
    for k in strikes:
        opt_k = option.__class__(
            K=k, option_type=option.option_type,
            maturity=option.maturity, option_class=option.option_class,
        )
        bs.update(K=k)
        k_vals.append(k)
        bs_vals.append(bs.price())
        tree_vals.append(tree.price(opt_k, build_tree=True))

    # --- graphique ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_vals, bs_vals, label="Black–Scholes", lw=2, color="steelblue")
    ax.scatter(k_vals, tree_vals, label="Trinomial Tree", s=25, color="darkorange")
    ax.set(title=f"Prix vs Strike (N={n_steps})", xlabel="Strike K", ylabel="Prix")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def bs_convergence_by_step(
    market, option, max_n=400, step=25, pruning=True, epsilon=1e-7
):
    """
    Étudie la convergence du trinomial vers Black–Scholes en fonction de N.
    """
    pricing_date = dt.datetime.today()
    T = (option.maturity - pricing_date).days / 365

    bs = BlackScholesPricer(
        S=market.S0, K=option.K, T=T, r=market.r, sigma=market.sigma,
        option_type=option.option_type,
        dividend=getattr(market, "dividend", 0.0),
        dividend_date=getattr(market, "dividend_date", None),
    )
    bs_price = bs.price()

    n_vals = np.arange(step, max_n + 1, step, dtype=int)
    tree_prices = [
        TrinomialTree(
            market, N=n, pruning=pruning,
            epsilon=epsilon, pricingDate=pricing_date,
        ).price(option, build_tree=True)
        for n in n_vals
    ]
    abs_errors = np.abs(np.array(tree_prices) - bs_price)

    # --- prix vs N ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(n_vals, tree_prices, color="darkorange", label="Trinomial Tree")
    ax1.axhline(bs_price, color="steelblue", ls="--", label="Black–Scholes")
    ax1.set(title="Convergence du prix vs N", xlabel="N", ylabel="Prix")
    ax1.legend(loc="upper left")
    fig1.tight_layout()
    plt.show()

    # --- erreur log ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(n_vals, abs_errors, color="crimson")
    ax2.set_yscale("log")
    ax2.set(title="Erreur absolue (échelle log)", xlabel="N", ylabel="|Erreur|")
    fig2.tight_layout()
    plt.show()
