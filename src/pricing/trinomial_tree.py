import numpy as np
import datetime as dt
from pricing import BlackScholesPricer
from node import Node
from model import Model

class TrinomialTree(Model):
    def __init__(self, market, option, N, pricingDate=None):
        self.market = market
        self.option = option
        self.N = N
        self.delta_t = option.T / N
        self.alpha = np.exp(market.sigma * np.sqrt(3 * self.delta_t))
        super().__init__(pricingDate if pricingDate else dt.datetime.today())

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

        # vérification des probabilités
        self.check_probability(self.p_down, "p_down")
        self.check_probability(self.p_up, "p_up")
        self.p_mid = 1 - self.p_up - self.p_down

        # mémorise le dernier noeud du milieu créé (utile pour le pricing)
        self.last_mid = None

        # rajout de la construction de l'arbre
        self.root = self._build_tree()

    def _link_columns(self, prev_mid: Node, next_mid: Node) -> Node:
        """
        Relie la colonne t (centrée sur prev_mid) à la colonne t+1 (centrée sur next_mid).
        Retourne next_mid pour mettre à jour prev_mid dans _build_tree.
        """
        # centre
        prev_mid.next_mid = next_mid
        next_mid.prev_mid = prev_mid

        # diagonales du noeud centrale de la colonne t
        if next_mid.up: prev_mid.next_up = next_mid.up
        if next_mid.down: prev_mid.next_down = next_mid.down

        # on descend tout en bas de la colonne t
        t, t_1 = prev_mid, next_mid
        while t.down and t_1.down:
            t = t.down
            t_1 = t_1.down

        # on passe de t à t+1
        t, t_1 = t.up, t_1.up
        while t and t_1:
            t.next_mid = t_1; t_1.prev_mid = t
            if t_1.up: t.next_up = t_1.up
            if t_1.down: t.next_down = t_1.down
            t, t_1 = t.up, t_1.up

        return next_mid

    def _build_tree(self) -> Node:
        """
        On fait la méthode du remplissage verticale.
        On part du milieu et on va de haut en bas.
        """
        # racine
        # centre colonne t=0
        root = Node(S=self.market.S0, proba=1.0) 
        prev_mid = root

        for i in range(self.N):
            # créer le milieu de la colonne t+1
            mid_price = prev_mid.S * np.exp(self.market.r * self.delta_t)
            next_mid = Node(S=mid_price, proba=prev_mid.proba*self.p_mid)

            # partie haute
            current = next_mid
            for _ in range(i+1):
                up = Node(S=current.S * self.alpha, proba=current.proba*self.p_up)
                up.down = current
                current.up = up
                current = up

            # partie basse
            current = next_mid
            for _ in range(i+1):
                down = Node(S=current.S / self.alpha, proba=current.proba*self.p_down)
                down.up = current
                current.down = down
                current = down

            # relie les deux colonnes
            prev_mid = self._link_columns(prev_mid, next_mid)

        self.last_mid = prev_mid
        root.tree = self
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