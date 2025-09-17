import numpy as np

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

class TrinomialTree:
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

    def price_option(self):
        """
        Calcule le prix de l’option avec backward induction vectorisée.
        """
        N = self.N
        dt = self.delta_t
        discount = np.exp(-self.market.r * dt)

        # --- Construction des prix du sous-jacent à maturité ---
        # Les états distincts vont de -N à +N (2N+1 états)
        j = np.arange(-N, N+1)
        S_T = self.market.S0 * (self.alpha**j)

        # Payoff terminal
        if self.option.option_type == "call":
            V = np.maximum(0, S_T - self.option.K)
        else:  # put
            V = np.maximum(0, self.option.K - S_T)

        # --- Backward induction ---
        for _ in range(N, 0, -1):
            # À l'étape t, il y a 2t+1 états possibles
            # On combine les valeurs des enfants (up, mid, down)
            V_next = discount * (
                self.p_up * V[2:] +
                self.p_mid * V[1:-1] +
                self.p_down * V[:-2]
            )
            V = V_next  # devient la valeur au temps précédent

        # Prix de l’option = valeur au nœud racine
        return V[0]

    def display_tree(self, STs):
        """
        Affiche l'arbre trinomial avec les prix à chaque nœud.
        :param STs: Liste des niveaux de l'arbre contenant les prix des nœuds.
        """

        G = nx.DiGraph()
        pos = {}

        for level, nodes in enumerate(STs):
            for i, price in enumerate(nodes):
                node_id = f"{level}-{i}"
                G.add_node(node_id, price=price)
                pos[node_id] = (level, -i)

                if level > 0:
                    parent_level = STs[level - 1]
                    parent_id = f"{level - 1}-{i // 3}"
                    G.add_edge(parent_id, node_id)

        labels = {node: f"{data['price']:.2f}" for node, data in G.nodes(data=True)}
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color="lightblue")
        plt.title("Arbre Trinomial des Prix")
        plt.show()