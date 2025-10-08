import numpy as np
import datetime as dt
from node import Node
from model import Model
from funcs import compute_forward, compute_variance, compute_probabilities
from typing import *

class TrinomialTree(Model):
    def __init__(
        self, market, option, N, pruning: bool = False, epsilon=None, pricingDate=None
    ):
        self.market = market
        self.option = option
        self.N = N
        if epsilon is not None:
            self.epsilon = epsilon
        self.pruning = pruning
        super().__init__(pricingDate if pricingDate else dt.datetime.today())

    def price(self, build_tree: bool = False) -> float:

        # pricing à partir de la racine
        if build_tree or not hasattr(self, "root") or self.root is None:
            self._compute_parameters(self.market.S0, dividend=False)
            self.root = self._build_tree()

        return self.root.price_backward(self.option)
    
    def _build_tree(self) -> Node:
        """
        Grille uniforme. Probas constantes (calibrées une fois dans price()).
        Au pas ex-div, on shift toute la colonne de -D puis on poursuit.
        """
        root = Node(S=self.market.S0, proba=1.0)
        t = root

        div_step = self._compute_div_step()

        for i in range(self.N):
            step_left = self.N - (i + 1)
            dividend = (div_step==step_left and self.market.dividend > 0)

            trunc_node = t

            if dividend:
                # choisir le nœud ancre (nearest) dans l'esperance
                t1 = self._create_next_trunc(t)
            else:
                t1 = self._create_next_mid(t)
            
            trunc_node_t_1 = t1
            self._extend_upper_part(t, t1, i)

            t = trunc_node; t1 = trunc_node_t_1
            self._extend_lower_part(t, t1, i)

            t = trunc_node_t_1

            if dividend:
                # on revient au schéma standard après le pas ex-div
                self._compute_parameters(t.S)

        root.tree = self
        return root

    def _create_next_mid(self, prev_mid: Node) -> Node:
        """Crée le noeud central de la colonne t+1 à partir de prev_mid (colonne t)."""
        mid_price = prev_mid.S * np.exp(self.market.r * self.delta_t)
        next_mid = Node(S=mid_price, proba=prev_mid.proba * self.p_mid)
        next_mid.tree = self

        # auto rattachement au centre
        next_mid.trunc = next_mid; next_mid.prev_trunc = prev_mid
        return next_mid

    def _extend_upper_part(self, t: Node, t1: Node, i: int) -> None:
        """Construit la partie haute de la colonne t et t+1 à partir du milieu.
        Ajout du lien next_down (si déjà disponible) vers le noeud down de la colonne t+1.
        Ce lien sera complété après la construction de la partie basse."""
        # rajout du trunc pour le pricing
        trunc_t = t; trunc_t1 = t1

        for _ in range(i + 1):
            # creation du prochain noeud up
            t1_s_up = t1.S * self.alpha; t1_p_up = t1.proba * self.p_up
            up = Node(S=t1_s_up, proba=t1_p_up); up.tree = self

            # rattachament à la base
            up.trunc = trunc_t1; up.prev_trunc = trunc_t
            
            t.next_mid = up.down = t1; t.next_up = t1.up = up
            # ajout du lien next_down (sera None tant que la partie basse n'est pas construite)
            t.next_down = t1.down

            t, t1 = t.up, t1.up
    
    def _extend_lower_part(self, t: Node, t1: Node, i: int) -> None:
        """Construit la partie basse de la colonne t+1 à partir du milieu, et complète les next_down manquants en haut."""
        # rajout du trunc pour le pricing
        trunc_t = t; trunc_t1 = t1
        for _ in range(i + 1):
            t.next_mid = t1; t.next_up = t1.up
            
            t1_s_down = t1.S / self.alpha; t1_p_down = t1.proba * self.p_down
            
            down = Node(S=t1_s_down, proba=t1_p_down); down.tree = self; down.up = t1
            
            # rattachament à la base
            down.trunc = trunc_t1; down.prev_trunc = trunc_t
            
            t.next_down = t1.down = down
            # si le noeud juste au-dessus n'a pas encore son next_down (ajouté en upper mais None), on le met à jour
            if t.up and (t.up.next_down is None):
                t.up.next_down = t1.down
            t, t1 = t.down, t1.down


    def _compute_parameters(self, S: float, S1_mid: float = None, dividend: bool = False) -> None:
        self.delta_t = (self.option.maturity - self.pricing_date).days / self.N / 365

        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.delta_t))

        forward  = compute_forward(S, self.market.r, self.delta_t)
        esperance = forward - self.market.dividend if dividend else forward
        variance = compute_variance(S, self.market.r, self.delta_t, self.market.sigma)

        self.p_down, self.p_up, self.p_mid = compute_probabilities(
            esperance, S1_mid if dividend else forward, variance, self.alpha, dividend
        )
        self._assert_probabilities(self.p_down, self.p_up, self.p_mid)

    def _assert_probabilities(self, p_down, p_up, p_mid):
        self.check_probability(p_down, "p_down")
        self.check_probability(p_up, "p_up")
        self.check_probability(p_mid, "p_mid")
        if not np.isclose(p_down + p_mid + p_up, 1.0, atol=1e-12):
            raise AssertionError("La somme des probabilités est différente de 1.")

    def _compute_div_step(self):
        """Calcule l'index d'étape du versement de dividende, ou None s'il n'y en a pas."""
        if not getattr(self.market, "dividend_date", None) or not getattr(self.market, "dividend", 0):
            return None
        years = (self.market.dividend_date - self.pricing_date).days / 365
        div_step = int(np.floor(years / self.delta_t + 1e-12))
        return max(1, min(div_step, self.N))

    def _create_next_trunc(self, node: Node) -> Node:
        """
        Crée le noeud le plus proche de l'espérance comme centre du prochain triplet.
        """
        esperance = compute_forward(node.S, self.market.r, self.delta_t) - self.market.dividend
        # point de départ : le mid sans dividende
        s_mid = node.S * np.exp(self.market.r * self.delta_t)

        # on ajuste la grille pour encadrer l'espérance
        while esperance < s_mid / self.alpha:
            s_mid /= self.alpha
        while esperance > s_mid * self.alpha:
            s_mid *= self.alpha

        # crée le nœud central avec ce mid
        next_mid = Node(S=s_mid, proba=node.proba * self.p_mid)
        next_mid.tree = self

        # Recalcule les paramètres à partir du bon S_mid
        self._compute_parameters(node.S, s_mid, dividend=True)

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
            bound = S * (self.alpha**step_left)
            european_bound = discount * max(bound - self.option.K, 0.0)
        else:
            bound = S / (self.alpha**step_left)
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
