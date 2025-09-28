import numpy as np

class Node:
    def __init__(self, S, proba, up=None, down=None, next_up=None, next_mid=None, next_down=None, prev_mid=None, option_value=None):
        self.S = S      # Prix du sous-jacent au nœud
        self.proba = proba  # Probabilité d'atteindre ce nœud
        self.up = up    # Nœud "up" frère
        self.down = down# Nœud "down" frère
        self.next_up = next_up    # Nœud "up" prochain
        self.next_mid = next_mid  # Nœud "mid" prochain
        self.next_down = next_down# Nœud "down" prochain
        self.prev_mid = prev_mid  # Nœud "mid" précédentid
        self.option_value = option_value  # Valeur de l'option au nœud
        self.tree = None
        
        
        # il faudra rajouter option value ici pour le pricing
        # on doit pouvoir faire tree.root.price(option)
    def price(self, option = None):
        """
        Calcule le prix de l’option en utilisant l’arbre trinomial.
        """
        if option is None:
            option = self.option

        # on calcule les payoffs aux feuilles
        node = self.tree.last_mid
        while node:
            node.option_value = option.payoff(node.S)
            node = node.up

        node = self.tree.last_mid.down
        while node:
            node.option_value = option.payoff(node.S)
            node = node.down

        node = self.tree.last_mid
        # on envoie la valeur aux nœuds parents
        for i in range(self.tree.N):
            current = node
            n=0
            if node.prev_mid is not None:  # pas une feuille
                while current.next_up is not None and current.next_down is not None and n<=i:
                    expected_value = (
                    self.tree.p_up * current.next_up.option_value +
                    self.tree.p_mid * current.next_mid.option_value +
                    self.tree.p_down * current.next_down.option_value
                    ) * np.exp(-self.tree.market.r * self.tree.delta_t)

                    # valeur au noeud courant
                    current.option_value = expected_value

                    # on monte d'un noeud
                    current = current.up

                    # incrément pour ne passer au noeud du dessus alors qu'il n'existe pas
                    n += i

                n=0
                # on restart au milieu
                while current.next_up is not None and current.next_down is not None and n<=i:
                    expected_value = (
                    self.tree.p_up * current.next_up.option_value +
                    self.tree.p_mid * current.next_mid.option_value +
                    self.tree.p_down * current.next_down.option_value
                    ) * np.exp(-self.tree.market.r * self.tree.delta_t)

                    # valeur au noeud courant
                    current.option_value = expected_value

                    # on monte d'un noeud
                    current = current.down

                    # incrément pour ne passer au noeud du dessus alors qu'il n'existe pas
                    n += i

            node = node.prev_mid  # remonte d’une colonne

        return self.root.option_value