import numpy as np
import datetime as dt
from node import Node
from model import Model
from funcs import (compute_forward
                   , compute_variance
                   , compute_p_down
                   , compute_p_up
                   , compute_p_mid)

class TrinomialTree(Model):
    def __init__(self, market, option, N, pruning: bool = False, epsilon = None, pricingDate=None):
        self.market = market
        self.option = option
        self.N = N
        if epsilon is not None:
            self.epsilon = epsilon
        self.pruning = pruning
        super().__init__(pricingDate if pricingDate else dt.datetime.today())

    def _compute_parameters(self) -> None:
        # on calcule delta t
        self.delta_t = (self.option.maturity - self.pricing_date).days / self.N / 365

        # calcul de alpha
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.delta_t))

        # Probabilités
        forward = compute_forward(self.market.S0, self.market.r, self.delta_t)
        variance = compute_variance(self.market.S0, self.market.r
                                      , self.delta_t
                                      , self.market.sigma)

        self.p_down = compute_p_down(forward, forward, variance, self.alpha)
        self.p_up = compute_p_up(self.p_down, self.alpha)

        # vérification des probabilités
        self.check_probability(self.p_down, "p_down")
        self.check_probability(self.p_up, "p_up")
        self.p_mid = compute_p_mid(self.p_down, self.p_up)

    def price(self, build_tree: bool = False) -> float:

        # pricing à partir de la racine
        if build_tree or not hasattr(self, "root") or self.root is None:
            self._compute_parameters()
            self.root = self._build_tree()
        
        return self.root.price_recursive(self.option)

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
            # gestion des etapes restantes pour le filtrage
            step_left = self.N - (i+1)

            # créer le milieu de la colonne t+1
            mid_price = prev_mid.S * np.exp(self.market.r * self.delta_t)
            next_mid = Node(S=mid_price, proba=prev_mid.proba * self.p_mid)
            # connaissance de l'arbre
            next_mid.tree = self

            # partie haute
            current = next_mid
            for _ in range(i + 1):
                # futures attributs du noeud up
                s_up = current.S * self.alpha
                p_up = current.proba * self.p_up

                # gestion du filtre par contribution au prix
                if self.pruning and self.__should_filter(s_up, step_left):
                    break

                up = Node(S=s_up, proba=p_up)
                # connaissance de l'arbre
                up.tree = self
                up.down = current
                current.up = up
                current = up

            # partie basse
            current = next_mid
            for _ in range(i + 1):
                # futures attributs du noeud down
                s_down = current.S / self.alpha
                p_down = current.proba * self.p_down

                # gestion du filtre par contribution au prix
                if self.pruning and self.__should_filter(s_down, step_left):
                    break

                down = Node(S=s_down, proba=p_down)
                # connaissance de l'arbre
                down.tree = self
                down.up = current
                current.down = down
                current = down

            # relie les deux colonnes
            prev_mid = self._link_columns(prev_mid, next_mid)

        root.tree = self
        return root

    def _link_columns(self, prev_mid: Node, next_mid: Node) -> Node:
        """
        Relie la colonne t (centrée sur prev_mid) à la colonne t+1 (centrée sur next_mid).
        Retourne next_mid pour mettre à jour prev_mid dans _build_tree.
        """
        # centre
        prev_mid.next_mid = next_mid

        # diagonales du noeud centrale de la colonne t
        if next_mid.up:
            prev_mid.next_up = next_mid.up
        if next_mid.down:
            prev_mid.next_down = next_mid.down

        # on descend tout en bas de la colonne t
        t, t_1 = prev_mid, next_mid
        while t.down and t_1.down:
            t = t.down
            t_1 = t_1.down

        # on passe de t à t+1
        while t and t_1:
            t.next_mid = t_1
            t.next_up = t_1.up if t_1.up else None
            t.next_down = t_1.down if t_1.down else None
            t, t_1 = t.up, t_1.up

        return next_mid
    
    def __upper_filter(self, S, step_left):
        """
        On va venir filtrer les noeuds qui ont une contribution faible à partir d'une
        valeur de epsilon defini en amont.
        """

        # dernière colonne
        if step_left <= 0:
            return self.option.payoff(S)
        
        discount = np.exp(-self.market.r * step_left * self.delta_t)

        # borne différentes si call/put
        if self.option.option_type == "call":
            bound = S * (self.alpha ** step_left)
            european_bound = discount * max(bound - self.option.K, 0.0)
        else:
            bound = S / (self.alpha ** step_left)
            european_bound = max(self.option.K - bound, 0.0)
        
        # gestion option americaine
        if self.option.option_class == "american":
            return max(self.option.payoff(S), european_bound)
        return european_bound
    
    def __should_filter(self, S, step_left):
        if self.epsilon is None:
            return False
        bound = self.__upper_filter(S, step_left)
        return bound < self.epsilon