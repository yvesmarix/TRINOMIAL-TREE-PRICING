import pandas as pd
from pricing import BlackScholesPricer, TrinomialTree

def bs_convergence_by_step(
    market, option, bs_price: float, max_n: int = 1000, step: int = 10
):
    """
    Calcule les prix de l’option pour différentes valeurs de N
    afin d’observer la convergence vers le modèle de Black-Scholes.
    """
    prices = []
    N_values = list(range(step, max_n + 1, step))
    for N in N_values:
        tree = TrinomialTree(market, option, N=N, pruning=True, epsilon=1e-7)
        prices.append(tree.price(option))

    price_dataset = pd.DataFrame(
        {
            "N": N_values,
            "Trinomial Price": prices,
            "Black-Scholes Price": [bs_price] * len(N_values),
        }
    )

    # Set 'N' as the index
    price_dataset.set_index("N", inplace=True)
    price_dataset.plot(
        title="Convergence du prix de l’option vers Black-Scholes",
        ylabel="Prix de l’option",
    )

def bs_convergence_by_strike(market, option, K_values: list, N: int):
    """
    Calcule les prix de l’option pour différentes valeurs de K
    afin d’observer la convergence vers le modèle de Black-Scholes.
    """
    tree_prices = []
    bs_prices = []
    for K in K_values:
        option.K = K
        tree = TrinomialTree(market, option, N=N)
        tree_prices.append(tree.price(option))
        bs_prices.append(
            BlackScholesPricer().price(
                S=market.S0, 
                K=option.K, 
                T=tree.delta_t*tree.N, 
                r=market.r, 
                sigma=market.sigma, 
                option_type='call', 
                dividend=market.dividend, 
                dividend_date = market.dividend_date
            )
        )  # q=0 car pas de dividendes

    price_dataset = pd.DataFrame(
        {
            "Strike Price (K)": K_values,
            "Trinomial Price": tree_prices,
            "Black-Scholes Price": bs_prices,
        }
    )

    # Set 'Strike Price (K)' as the index
    price_dataset.set_index("Strike Price (K)", inplace=True)
    price_dataset.plot(
        title="Convergence du prix de l’option vers Black-Scholes",
        ylabel="Prix de l’option",
    )
