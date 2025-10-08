import numpy as np
from pricing.funcs import iter_column

class Node:

    def __init__(
        self,
        S,
        proba,
        up=None,
        down=None,
        next_up=None,
        next_mid=None,
        next_down=None,
        trunc=None,
        prev_trunc=None,
        option_value=None,
    ):
        self.S = S  # Prix du sous-jacent au nœud
        self.proba = proba  # Probabilité d'atteindre ce nœud
        self.up = up  # Nœud "up" frère
        self.down = down  # Nœud "down" frère
        self.next_up = next_up  # Nœud "up" prochain
        self.next_mid = next_mid  # Nœud "mid" prochain
        self.next_down = next_down  # Nœud "down" prochain
        self.trunc=trunc
        self.prev_trunc=prev_trunc
        self.option_value = option_value  # Valeur de l'option au nœud
        self.tree = None

        # il faudra rajouter option value ici pour le pricing
        # on doit pouvoir faire tree.root.price(option)
    
    def price_recursive(self, option) -> float:
        """
        Calcule la valeur de l'option à ce nœud de façon récursive.
        Empêche la récursion infinie en mémorisant les valeurs déjà calculées.
        Utilise un ensemble 'visited' pour éviter les cycles.
        """
        # Si la valeur de l'option a déjà été calculée, on la retourne
        if self.option_value is not None:
            return self.option_value

        # payoffs aux feuilles
        if self.next_up is None and self.next_mid is None and self.next_down is None:
            self.option_value = option.payoff(self.S)
            return self.option_value
        
        # calcul récursif des enfants
        v_up = self.tree.p_up * (self.next_up.price_recursive(option) if self.next_up else 0.0)
        v_mid = self.tree.p_mid * (self.next_mid.price_recursive(option) if self.next_mid else 0.0)
        v_down = self.tree.p_down * (self.next_down.price_recursive(option) if self.next_down else 0.0)

        # actualisation
        discount = np.exp(-self.tree.market.r * self.tree.delta_t)
        continuation = discount * (v_up + v_mid + v_down)

        if option.option_class == "american":
            exercise = option.payoff(self.S)
            self.option_value = max(exercise, continuation)
        else:
            self.option_value = continuation
        return self.option_value
    
    def price(self, option) -> float:
        # calcul des enfants
        v_up = self.tree.p_up * (self.next_up.option_value if self.next_up else 0.0)
        v_mid = self.tree.p_mid * (self.next_mid.option_value if self.next_mid else 0.0)
        v_down = self.tree.p_down * (self.next_down.option_value if self.next_down else 0.0)

        # actualisation
        discount = np.exp(-self.tree.market.r * self.tree.delta_t)
        continuation = discount * (v_up + v_mid + v_down)

        if option.option_class == "american":
            exercise = option.payoff(self.S)
            self.option_value = max(exercise, continuation)
        else:
            self.option_value = continuation
        
        return self.option_value
        

    def price_backward(self, option=None):
        """
        Backward induction explicite (sans récursion) avec factorisation.
        1. Trouve la dernière colonne via next_mid.
        2. Initialise les payoffs aux feuilles.
        3. Remonte colonne par colonne via prev_trunc.
        """
        if option is None:
            raise ValueError("Option requise pour le pricing backward.")

        # aller jusqu'à la dernière colonne (chaîne des next_mid)
        last_mid = self.tree.root
        while last_mid.next_mid:
            last_mid = last_mid.next_mid

        # initialisation des payoffs aux feuilles
        for leaf in iter_column(last_mid):
            leaf.option_value = option.payoff(leaf.S)

        p_up = self.tree.p_up
        p_mid = self.tree.p_mid
        p_down = self.tree.p_down
        discount = np.exp(-self.tree.market.r * self.tree.delta_t)
        is_american = (option.option_class == "american")
        payoff = option.payoff

        def compute_node(n):
            if n.option_value is not None:
                return
            vu = p_up * (n.next_up.option_value if n.next_up else 0.0)
            vm = p_mid * (n.next_mid.option_value if n.next_mid else 0.0)
            vd = p_down * (n.next_down.option_value if n.next_down else 0.0)
            cont = discount * (vu + vm + vd)
            if is_american:
                ex = payoff(n.S)
                n.option_value = ex if ex > cont else cont
            else:
                n.option_value = cont

        # remontée colonne par colonne
        mid = last_mid
        while mid.prev_trunc:
            prev_mid = mid.prev_trunc
            for n in iter_column(prev_mid):
                compute_node(n)
            mid = prev_mid

        return self.tree.root.option_value
