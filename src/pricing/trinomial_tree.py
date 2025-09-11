import numpy as np

class TrinomialTreeNode:
    def __init__(self, price, time, up=None, mid=None, down=None):
        self.price = price
        self.time = time
        self.up = up
        self.mid = mid
        self.down = down
        self.option_value = None

class TrinomialTree:
    def __init__(self, S0, r, sigma, T, N, K, D=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.K = K
        self.D = D if D is not None else [0] * N
        self.delta_t = T / N # pas de temps egaux
        self.alpha = np.exp(sigma * np.sqrt(3 * self.delta_t)) # choix arbitraire de sqrt(3) 
        self.root = None

    def _calculate_probabilities(self, Sti):
        """
        Determine les probabilités de transitions vers chaque noeud à
        partir du systeme d'equation suivant : 
        - à chaque noeud la somme des 3 probabilités p_up + p_down + p_mid = 1
        - avec comme condition que la moyenne pondérée des S_up/mid/down par leurs 
            probabilités respectives soit égale à l'espérance du prix en i+1 
            conditionnelle au prix en i qui vaut = Sti*exp(r*delta de temps) - Dti+1
        - avec comme autre condition que la variance 
        soit = (Sti^2)*(exp(2r*delta de temps))*(exp(sigma^2 * delta de temps)-1)
        """
        p_down = (np.exp(self.sigma**2 * self.delta_t) - 1) / ((1 - self.alpha) * (self.alpha**(-2) - 1))
        p_up = p_down / self.alpha
        p_mid = 1 - p_up - p_down
        return p_up, p_mid, p_down

    def _build_tree(self):
        nodes = {}
        # initialisation de l'arbre
        nodes[(0, 0)] = TrinomialTreeNode(self.S0, 0) # racine de l'arbre

        for i in range(1, self.N + 1):
            # j peut prendre la valeur -i, 0 ou i pour les 3 etats possibles down/mid/up
            for j in range(-i, i + 1):
                # calcul du prix forward
                if i == 1:
                    S_mid = self.S0 * np.exp(self.r * self.delta_t) - self.D[0]
                else:
                    parent_j = j
                    parent_node = nodes[(i-1, parent_j)]
                    # forward ajuste des dividendes
                    S_mid = parent_node.price * np.exp(self.r * self.delta_t) - self.D[i-1]

                S_up = S_mid * self.alpha
                S_down = S_mid / self.alpha

                # creation des noeuds
                nodes[(i, j)] = TrinomialTreeNode(S_up, i * self.delta_t)
                nodes[(i, j-1)] = TrinomialTreeNode(S_mid, i * self.delta_t)
                nodes[(i, j-2)] = TrinomialTreeNode(S_down, i * self.delta_t)

                # lien avec le parent
                parent_node.up = nodes[(i, j)]
                parent_node.mid = nodes[(i, j-1)]
                parent_node.down = nodes[(i, j-2)]

        self.root = nodes[(0, 0)]
        return nodes

    def _evaluate_payoff(self, node):
        if node.price > self.K:
            return node.price - self.K
        else:
            return 0

    def _backward_induction(self, nodes):
        for i in range(self.N, 0, -1):
            for j in range(-i, i + 1):
                node = nodes[(i, j)]
                if node is None:
                    continue
                node.option_value = self._evaluate_payoff(node)

            for j in range(-i, i + 1):
                node = nodes[(i-1, j)]
                if node is None or node.up is None:
                    continue
                p_up, p_mid, p_down = self._calculate_probabilities(node.price)
                node.option_value = np.exp(-self.r * self.delta_t) * (
                    p_up * node.up.option_value +
                    p_mid * node.mid.option_value +
                    p_down * node.down.option_value
                )

        return self.root.option_value

    def price_option(self):
        nodes = self._build_tree()
        return self._backward_induction(nodes)

