import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from pricing import BlackScholesPricer, TrinomialTree

def bs_convergence_by_strike(
    market, option, strikes, n_steps=200, pruning=True, epsilon=1e-7
):

    pricing_date = dt.datetime.today()
    time_to_maturity = (option.maturity - pricing_date).days / 365

    # Instanciation OOP du pricer BS (sans repasser les args à price)
    bs = BlackScholesPricer(
        S=market.S0,
        K=option.K,
        T=time_to_maturity,
        r=market.r,
        sigma=market.sigma,
        option_type=option.option_type,
        dividend=getattr(market, "dividend", 0.0),
        dividend_date=getattr(market, "dividend_date", None),
    )

    # Instanciation Tree
    tree = TrinomialTree(
    market,
    N=n_steps,
    pruning=pruning,
    epsilon=epsilon,
    pricingDate=pricing_date,
    )

    k_values, bs_values, tree_values = [], [], []
    for k in strikes:
        # Option clonée avec strike k pour l’arbre
        opt_k = option.__class__(
            K=k,
            option_type=option.option_type,
            maturity=option.maturity,
            option_class=option.option_class,
        )

        tree_price = tree.price(opt_k, build_tree=True)

        # BS avec K mis à jour (OOP)
        bs.update(K=k)
        bs_price = bs.price()

        k_values.append(k)
        bs_values.append(bs_price)
        tree_values.append(tree_price)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, bs_values, color="steelblue", linewidth=2, label="Black–Scholes")
    ax.scatter(k_values, tree_values, color="darkorange", s=25, label="Trinomial Tree")
    ax.set_title(f"Prix vs Strike (N={n_steps})")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Prix")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def bs_convergence_by_step(
    market, option, max_n=400, step=25, pruning=True, epsilon=1e-7
):
    pricing_date = dt.datetime.today()
    time_to_maturity = (option.maturity - pricing_date).days / 365

    # Instanciation OOP du pricer BS
    bs = BlackScholesPricer(
        S=market.S0,
        K=option.K,
        T=time_to_maturity,
        r=market.r,
        sigma=market.sigma,
        option_type=option.option_type,
        dividend=getattr(market, "dividend", 0.0),
        dividend_date=getattr(market, "dividend_date", None),
    )
    bs_price = bs.price()

    n_values = np.arange(step, max_n + 1, step, dtype=int)
    tree_prices = [
        TrinomialTree(
            market,
            N=n,
            pruning=pruning,
            epsilon=epsilon,
            pricingDate=pricing_date,
        ).price(option, build_tree=True)
        for n in n_values
    ]
    abs_errors = np.abs(np.array(tree_prices) - bs_price)

    # --- Prix vs N ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(n_values, tree_prices, color="darkorange", label="Trinomial Tree")
    ax1.hlines(
        bs_price,
        n_values.min(),
        n_values.max(),
        colors="steelblue",
        linestyles="dashed",
        label="Black–Scholes",
    )
    ax1.set_title("Convergence du prix vs N")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Prix")
    ax1.legend(loc="upper left")
    fig1.tight_layout()
    plt.show()

    # --- Erreur absolue ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(n_values, abs_errors, color="crimson")
    ax2.set_yscale("log")
    ax2.set_title("Erreur absolue (échelle log)")
    ax2.set_xlabel("N")
    ax2.set_ylabel("|Erreur|")
    fig2.tight_layout()
    plt.show()
