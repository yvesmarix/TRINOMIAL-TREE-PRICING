import numpy as np
import datetime as dt
from node import Node
from model import Model
from funcs import compute_forward, compute_variance, compute_probabilities, iter_column
from typing import *

class TrinomialTree(Model):
    """
    Arbre trinomial (construction « verticale » colonne par colonne).

    Idées directrices:
    - On stocke dans chaque Node: S (spot), proba (masse cumulée d'arriver là),
      liens verticaux (up/down) + liens horizontaux (next_* vers colonne t+1).
    - On construit la colonne t+1 à partir du noeud central (mid) puis on
      « grimpe » vers le haut et on « descend » vers le bas.
    - Pruning (optionnel): si la probabilité totale future d'un up/down potentiel
      est trop petite (< epsilon) on ne crée pas la branche; on redirige la masse
      vers le mid => transition monomiale (variance réduite localement).
    - Dividende: à l’étape ex-div on recalibre la position du mid autour
      de l’espérance forward - D (en déplaçant le mid si besoin) puis on
      recalcule les probabilités avec ce nouveau centre.

    Hypothèses:
    - Probabilités (p_down, p_mid, p_up) constantes entre deux dates de dividende.
    - price_backward du Node sait traiter l’absence de certaines branches
      (sinon prévoir garde-fous).
    """

    def __init__(
        self, market, option, N, pruning: bool = False, epsilon=None, pricingDate=None
    ):
        self.market = market
        self.option = option
        self.N = N
        # seuil absolu de pruning (probabilité minimale pour créer la branche)
        self.epsilon = epsilon if epsilon is not None else 1e-7
        self.pruning = pruning
        super().__init__(pricingDate if pricingDate else dt.datetime.today())

    def price(self, build_tree: bool = False) -> float:
        """
        Calcule le prix (backward induction).
        - (Re)construit l'arbre si nécessaire.
        - Retourne la valeur au noeud racine.
        """
        if build_tree or not hasattr(self, "root") or self.root is None:
            # Probabilités calibrées avant construction (colonne 0 -> 1)
            self._compute_parameters(self.market.S0, dividend=False)
            self.root = self._build_tree()
        return self.root.price_backward(self.option)
    
    def _build_tree(self) -> Node:
        root = Node(S=self.market.S0, proba=1.0)
        root.tree = self
        t = root

        div_step = self._compute_div_step()

        for i in range(self.N):
            step_left = self.N - (i + 1)
            dividend = (div_step == step_left and self.market.dividend > 0)

            # Crée les enfants pour chaque nœud de la colonne courante
            for node, is_bord in iter_column(t, mode="building"):
                node.create_children(self, i, pruning=self.pruning, is_bord=is_bord, dividend=dividend)

            # retour aux probas normales
            if dividend:
                self._compute_parameters(t.S)

            # Avance à la colonne suivante
            t = t.next_mid

        root.tree = self
        return root

    def _compute_parameters(self, S: float, S1_mid: float = None, dividend: bool = False) -> None:
        """
        Calibrage local des paramètres de la maille:
        - delta_t, alpha (écart multiplicatif up / down),
        - p_down, p_mid, p_up via compute_probabilities.
        - Si dividend=True: l'espérance est forward - D et S1_mid est le mid choisi.
        """
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
        """
        Vérifie: chaque proba dans [0,1] et somme ≈ 1. Sinon on lève une AssertionError.
        """
        self.check_probability(p_down, "p_down")
        self.check_probability(p_up, "p_up")
        self.check_probability(p_mid, "p_mid")
        if not np.isclose(p_down + p_mid + p_up, 1.0, atol=1e-12):
            raise AssertionError("La somme des probabilités est différente de 1.")

    def _compute_div_step(self):
        """
        Calcule l'index d'étape où le dividende tombe (en partant de la fin).
        Retour:
        - None si pas de dividende configuré.
        - Entier dans [1, N] sinon.
        """
        if not getattr(self.market, "dividend_date", None) or not getattr(self.market, "dividend", 0):
            return None
        years = (self.market.dividend_date - self.pricing_date).days / 365
        div_step = int(np.floor(years / self.delta_t + 1e-12))
        return max(1, min(div_step, self.N))