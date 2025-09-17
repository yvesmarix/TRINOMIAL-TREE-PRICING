import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Market:
    def __init__(self, S0, r, sigma, D=None):
        self.S0 = S0      # Prix initial de l'actif sous-jacent
        self.r = r        # Taux d'intérêt sans risque
        self.sigma = sigma # Volatilité de l'actif sous-jacent
        self.D = D if D is not None else [] # Dividendes discrets

class Option:
    def __init__(self, K, option_type, T):
        self.K = K  # Strike price
        self.option_type = option_type
        self.T = T

class Node:
    def __init__(self, underlying, tree):
        self.underlying = underlying
        self.tree = tree
        self.node_up = None
        self.node_mid = None
        self.node_down = None

class TrinomialTree:
    def __init__(self, market, option, N):
        self.market = market
        self.option = option
        self.N = N
        self.delta_t = option.T / N
        self.alpha = np.exp(market.sigma * np.sqrt(3 * self.delta_t))
        self.root = None

        # calcul des probabilités une seule fois
        forward_1 = np.exp(self.market.r * self.delta_t)*self.market.S0
        variance_1 = self.market.S0**2 * np.exp(2*self.market.r*self.delta_t)*(np.exp(self.market.sigma**2 * self.delta_t)-1)

        # Calcul des probabilités
        self.p_down = (forward_1**(-2)*(variance_1+forward_1**2)-1-(self.alpha+1)*(forward_1**(-1)*forward_1-1))/((1-self.alpha)*(self.alpha**(-2)-1))
        self.p_up = self.p_down / self.alpha
        self.p_mid = 1 - self.p_up - self.p_down

    def build_tree(self):
        """
        Construit l'arbre trinomial en créant des objets Node à chaque étape.
        """
        self.root = Node(underlying=self.market.S0, tree=self)
        self.levels = [[self.root]]
        
        for _ in range(self.N):
            next_level = []
            for node in self.levels[-1]:
                up = node.underlying * self.alpha
                mid = node.underlying * np.exp(self.market.r * self.delta_t)
                down = node.underlying / self.alpha
                node_up = Node(underlying=up, tree=self)
                node_mid = Node(underlying=mid, tree=self)
                node_down = Node(underlying=down, tree=self)
                node.node_up = node_up
                node.node_mid = node_mid
                node.node_down = node_down
                next_level.extend([node_up, node_mid, node_down])
            self.levels.append(next_level)

        return self.root
    
    def price_option(self):
        """
        Calcule le prix de l'option en parcourant l'arbre et en utilisant les probabilités.
        Retourne le prix du call et du put.
        """
        if self.root is None:
            self.build_tree()

        # On commence par calculer les payoffs finaux pour chaque feuille
        leaves = self.levels[-1]
        match self.option.option_type:
            case "call":
                for leaf in leaves:
                    leaf.call_payoff = max(0, leaf.underlying - self.option.K)
            case "put":
                for leaf in leaves:
                    leaf.put_payoff = max(0, self.option.K - leaf.underlying)

        # On remonte l'arbre pour calculer les prix actualisés
        self._backward_induction(self.root)

        # Le prix de l'option est à la racine
        match self.option.option_type:
            case "call":
                return self.root.call_payoff
            case "put":
                return self.root.put_payoff

    def _backward_induction(self, node):
        """
        Effectue la remontée de l'arbre pour calculer les prix actualisés.
        """
        discount = np.exp(-self.market.r*self.delta_t)

        # On part de l'avant dernier niveau et on revient à la racine
        for level in reversed(self.levels[:-1]):
            for node in level:
                match self.option.option_type:
                    case "call":
                        node.call_payoff = discount * (
                            self.p_up * node.node_up.call_payoff +
                            self.p_mid * node.node_mid.call_payoff +
                            self.p_down * node.node_down.call_payoff
                        )
                    case "put":
                        node.put_payoff = discount * (
                            self.p_up * node.node_up.put_payoff +
                            self.p_mid * node.node_mid.put_payoff +
                            self.p_down * node.node_down.put_payoff
                        )


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