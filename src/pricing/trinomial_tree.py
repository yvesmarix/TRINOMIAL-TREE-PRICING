import numpy as np
import pandas as pd
from blackscholes import BlackScholesPricer

class Market:
    def __init__(self, S0, r, sigma, D=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.D = D if D is not None else []

class Option:
    def __init__(self, K, option_type, T):
        self.K = K
        self.option_type = option_type
        self.T = T

class Node:
    def __init__(self, S, proba, up, down, next_up, next_mid, next_down):
        self.S = S      # Prix du sous-jacent au nœud
        self.proba = proba  # Probabilité d'atteindre ce nœud
        self.up = up    # Nœud "up" frère
        self.down = down# Nœud "down" frère
        self.next_up = next_up    # Nœud "up" prochain
        self.next_mid = next_mid  # Nœud "mid" prochain
        self.next_down = next_down# Nœud "down" prochain
        # il faudra rajouter option value ici pour le pricing
        # il faut donner l'objet option à la racine de l’arbre
        # on doit pouvoir faire tree.root.price(option)
        # pour le pricing si next mid est None alors on est à la fin de l’arbre et on doit pricer

        # la valeur intrinsèque ou payoff doit être une methode de la classe option 
        # et on doit lui donner node.S pour pricer

class TrinomialTree:
    # ce serait bien de lui faire hériter de Model (voir moodle) une fonction check probability et str_pc, les deux en staticmethod
    def __init__(self, market, option, N):
        self.market = market
        self.option = option
        self.N = N
        self.delta_t = option.T / N
        self.alpha = np.exp(market.sigma * np.sqrt(3 * self.delta_t))

        # Probabilités (communes à tout l’arbre)
        forward_1 = np.exp(self.market.r * self.delta_t) * self.market.S0
        variance_1 = (
            self.market.S0**2 *
            np.exp(2 * self.market.r * self.delta_t) *
            (np.exp(self.market.sigma**2 * self.delta_t) - 1)
        )
        self.p_down = (
            (forward_1**(-2) * (variance_1 + forward_1**2) - 1 -
             (self.alpha+1) * (forward_1**(-1)*forward_1 - 1))
            / ((1 - self.alpha) * (self.alpha**(-2) - 1))
        )
        self.p_up = self.p_down / self.alpha
        self.p_mid = 1 - self.p_up - self.p_down

        # rajout de la construction de l'arbre
        self.root = self._build_tree()



    def _build_tree(self):

        # Créer le nœud racine
        root = Node(
            S=self.market.S0,
            proba=1.0,
            up=None,
            down=None,
            next_up=None,
            next_mid=None,
            next_down=None
        )

        current_node = root

        for i in range(self.N + 1):
            
            next_mid = Node(
                S=current_node.S*np.exp(self.market.r * self.delta_t),
                proba=current_node.proba * self.p_mid,
                up=None,
                down=None,
                next_up=None,
                next_mid=None,
                next_down=None
            )

            current_node.next_mid = next_mid
            current_node = next_mid

            n = 0
            while n<=i:
                up = Node(
                    S=current_node.S * self.alpha,
                    proba=current_node.proba * self.p_up,
                    up=None,
                    down=current_node,
                    next_up=current_node.next_mid.up if current_node.next_mid else None,
                    next_mid=None,
                    next_down=current_node.next_mid.down if current_node.next_mid else None
                )

                current_node.up = up
                current_node = up

                n += 1

            # on revient au milieu
            current_node = next_mid

            n=0
            while n<=i:
                down = Node(
                    S=current_node.S / self.alpha,
                    proba=current_node.proba * self.p_down,
                    up=current_node,
                    down=None,
                    next_up=None,
                    next_mid=None,
                    next_down=None
                )
                current_node.down = down
                current_node = down

                n += 1

        return root

    def bs_convergence_by_step(self, bs_price: float,max_n: int = 1000, step: int = 10):
        """
        Calcule les prix de l’option pour différentes valeurs de N
        afin d’observer la convergence vers le modèle de Black-Scholes.
        """
        prices = []
        N_values = list(range(step, max_n + 1, step))
        for N in N_values:
            self.N = N
            self.delta_t = self.option.T / N
            self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.delta_t))
            prices.append(self.price_option())
        
        price_dataset = pd.DataFrame({
            'N': N_values,
            'Trinomial Price': prices,
            'Black-Scholes Price': [bs_price] * len(N_values)
        })

        # Set 'N' as the index
        price_dataset.set_index('N', inplace=True)
        price_dataset.plot(title='Convergence du prix de l’option vers Black-Scholes', ylabel='Prix de l’option')

    def bs_convergence_by_strike(self, K_values: list):
        """
        Calcule les prix de l’option pour différentes valeurs de K
        afin d’observer la convergence vers le modèle de Black-Scholes.
        """
        tree_prices = []
        bs_prices = []
        for K in K_values:
            self.option.K = K
            tree_prices.append(self.price_option())
            bs_prices.append(BlackScholesPricer().price(S=self.market.S0, K=K, T=self.option.T, r=self.market.r, q=0, sigma=self.market.sigma, option_type=self.option.option_type)) # q=0 car pas de dividendes

        price_dataset = pd.DataFrame({
            'Strike Price (K)': K_values,
            'Trinomial Price': tree_prices,
            'Black-Scholes Price': bs_prices
        })

        # Set 'Strike Price (K)' as the index
        price_dataset.set_index('Strike Price (K)', inplace=True)
        price_dataset.plot(title='Convergence du prix de l’option vers Black-Scholes', ylabel='Prix de l’option')