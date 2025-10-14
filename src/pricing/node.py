import numpy as np
from pricing.funcs import iter_column, compute_forward

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

    def _find_nearest_node(self, tree: object, i: int):
        
        esperance = compute_forward(self.S, tree.market.r, tree.delta_t) - tree.market.dividend
        s_mid = self.S * np.exp(tree.market.r * tree.delta_t)

        # génère plusieurs candidats autour de s_mid
        candidates = []
        for k in range(-i-1, i):  # explore de alpha^-i à alpha^i
            candidate = s_mid * (tree.alpha ** k)
            candidates.append(candidate)
        # choisit le candidat le plus proche de l'espérance
        return min(candidates, key=lambda x: abs(x - esperance))

    def create_mid(self, tree, i: int, dividend: bool = False):
        """
        Crée le nœud 'mid' pour la prochaine colonne.

        Étape ex-div:
        - On veut que le mid (colonne t+1) soit le plus proche de l'espérance = forward - D.
        - On génère plusieurs candidats autour de s_mid (par puissances de alpha).
        - On choisit le candidat le plus proche de l'espérance.
        - Puis on recalcule les probabilités en mode dividend=True.
        """

        # pre-calculation
        proba_mid = self.proba * tree.p_mid
        s_mid = self._find_nearest_node(tree, i) if dividend else self.S * np.exp(tree.market.r * tree.delta_t)

        # check si deja existant
        if self.down is not None and self.down.next_up is not None:
            self.next_mid = self.down.next_up; self.down.next_up.proba += proba_mid
        elif self.up is not None and self.up.next_down is not None:
            self.next_mid = self.up.next_down; self.up.next_down.proba += proba_mid
        else: # creation
            self.next_mid = Node(S=s_mid, proba=proba_mid)
            self.next_mid.tree = self
            
            # rattachement central
            if self.trunc is None:
                self.trunc = self
            self.next_mid.trunc = self.next_mid; self.next_mid.prev_trunc = self.trunc

            # recalibration des probas sur le bon mid au pas ex-div
            if dividend:
                tree._compute_parameters(self.S, s_mid, dividend=True)


    def create_up(self, tree: object, pruning: bool, is_bord: bool):
        """
        Crée le nœud 'up' pour la prochaine colonne.
        """
        # pre-calculation
        proba_up = self.proba * tree.p_up

        # check si noeud deja existant
        if self.up is not None and self.up.next_mid is not None:
            self.next_up = self.up.next_mid; self.up.next_mid.proba += proba_up
        # pruning
        elif pruning and proba_up < tree.epsilon and is_bord:
            # Accumule la probabilité dans le mid si pruning pour eviter la perte de masse
            self.next_mid.proba += proba_up
        else:
            self.next_up = Node(S=self.next_mid.S * tree.alpha, proba=proba_up)
            self.next_up.tree = tree
            self.next_up.trunc = self.next_mid.trunc; self.next_up.prev_trunc = self.trunc

            # Ajout des liens verticaux
            self.next_mid.up = self.next_up; self.next_up.down = self.next_mid

    def create_down(self, tree: object, pruning: bool, is_bord: bool):
        """
        Crée le nœud 'down' pour la prochaine colonne.
        """
        # pre-calculation
        proba_down = self.proba * tree.p_down

        # check si noeud deja existant
        if self.down is not None and self.down.next_mid is not None:
            self.next_down = self.down.next_mid; self.down.next_mid.proba += proba_down
        # pruning
        elif pruning and proba_down < tree.epsilon and is_bord:
            # Accumule la probabilité dans le mid si pruning pour eviter la perte de masse
            self.next_mid.proba += proba_down
        else:
            self.next_down = Node(S=self.next_mid.S / tree.alpha, proba=proba_down)
            self.next_down.tree = tree; self.next_down.trunc = self.next_mid.trunc
            self.next_down.prev_trunc = self.trunc
            # Ajout des liens verticaux
            self.next_mid.down = self.next_down; self.next_down.up = self.next_mid

    def create_children(self, tree, i: int, pruning: bool = False, is_bord: bool = False, dividend: bool = False):
        """
        Crée les enfants (up, mid, down) pour ce nœud.
        - Utilise les méthodes spécialisées pour chaque branche.
        """
        # Crée le mid
        self.create_mid(tree, i, dividend)

        # Crée le up
        self.create_up(tree, pruning, is_bord)

        # Crée le down
        self.create_down(tree, pruning, is_bord)
    
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
