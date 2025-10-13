import numpy as np
import datetime as dt
from node import Node
from model import Model
from funcs import compute_forward, compute_variance, compute_probabilities
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
        """
        Construit l'arbre:
        - Démarre à la racine (S0).
        - Pour chaque pas:
            * Crée le mid de t+1 (ou mid ex-div ajusté).
            * Étend partie haute, puis partie basse.
            * Recalcule les paramètres juste après le dividende (pour revenir au régime standard).
        - Renseigne root.tree pour accès inverse.
        """
        root = Node(S=self.market.S0, proba=1.0)
        t = root

        div_step = self._compute_div_step()  # index de l'étape ex-div (ou None)

        for i in range(self.N):
            step_left = self.N - (i + 1)
            dividend = (div_step==step_left and self.market.dividend > 0)

            trunc_node = t  # on garde une ancre sur la ligne « centrale » précédente

            if dividend:
                # Choisir le mid le plus proche de l'espérance ex-div (forward - D)
                t1 = self._create_next_trunc(t, i)
            else:
                t1 = self._create_next_mid(t)
            
            trunc_node_t_1 = t1

            # Partie haute (on monte en parallèle dans les deux colonnes)
            self._extend_upper_part(t, t1, i)

            # Revenir au mid précédent pour descendre
            t = trunc_node; t1 = trunc_node_t_1
            self._extend_lower_part(t, t1, i)

            # Avancer la « colonne courante »
            t = trunc_node_t_1

            if dividend:
                # Après l'étape ex-div, on rétablit les probabilités standard
                self._compute_parameters(t.S)

        root.tree = self
        return root

    def _create_next_mid(self, prev_mid: Node) -> Node:
        """
        Crée le noeud central de la colonne t+1.
        - Prix forward simple : S * exp(r * dt)
        - Masse: proba(prev_mid) * p_mid
        """
        mid_price = prev_mid.S * np.exp(self.market.r * self.delta_t)
        next_mid = Node(S=mid_price, proba=0.0); next_mid.tree = self
        
        # rattachement central (utile pour backward)
        next_mid.trunc = next_mid; next_mid.prev_trunc = prev_mid
        next_mid.proba += prev_mid.proba * self.p_mid
        return next_mid

    def _extend_upper_part(self, t: Node, t1: Node, i: int) -> None:
        """
        Construit la partie haute des colonnes t (ancienne) et t+1 (nouvelle):
        - On part du mid et on itère vers le haut i+1 fois max.
        - À chaque niveau:
            * On calcule la proba totale future d'un potentiel noeud 'up'.
            * Pruning éventuel (redirige vers mid).
            * Sinon on instancie le noeud up et on relie.
        - Arrêt si plus de niveau disponible (None) ou pruning.
        """
        trunc_t = t; trunc_t1 = t1  # mémoires pour rattacher les truncs

        for _ in range(i + 1):
            contrib_mid, contrib_prev_up_mid, total_future_up = self._compute_branch_contributions(t, "up")

            # Décision de pruning (pas d'allocation si masse trop faible)
            if self._maybe_prune_branch(t, t1, "up", total_future_up):
                break

            # Allocation du noeud up
            up_price = t1.S * self.alpha; up = Node(S=up_price, proba=0.0)
            up.tree = self; up.trunc = trunc_t1; up.prev_trunc = trunc_t
            # Ajout contributions (masse venant du mid de t et du up-mid précédent)
            up.proba += contrib_mid + contrib_prev_up_mid

            # Liens horizontaux
            t.next_mid = t1; t.next_up = up; 

            # Chaînage vertical côté colonne t+1
            t1.up = up

            # Monter d'un cran
            t = t.up
            if t:
                t.next_down = t1
            t1 = t1.up
            if t is None or t1 is None:
                break

    
    def _extend_lower_part(self, t: Node, t1: Node, i: int) -> None:
        """
        Construit la partie basse des colonnes.
        - Symétrique de la partie haute mais en descendant.
        - On crée (ou réutilise) le noeud down lorsque nécessaire.
        - Pruning possible avant allocation (monomialisation).
        - On complète aussi les next_down manquants pour les noeuds au-dessus.
        """
        trunc_t = t; trunc_t1 = t1
        for _ in range(i + 1):
            contrib_mid, contrib_prev_down_mid, total_future_down = self._compute_branch_contributions(t, "down")

            # Pruning avant de créer le down (si la masse future est trop faible)
            pruned = False
            if t1.down is None and self._maybe_prune_branch(t, t1, "down", total_future_down):
                pruned = True
            if pruned:
                break

            # Création ou réutilisation du noeud down
            if t1.down is None:
                down_price = t1.S / self.alpha; down = Node(S=down_price, proba=0.0)
                down.tree = self; down.up = t1; down.trunc = trunc_t1
                down.prev_trunc = trunc_t; t1.down = down
            else:
                down = t1.down

            # Ajout des contributions
            down.proba += contrib_mid + contrib_prev_down_mid

            # Liens horizontaux
            t.next_mid = t1; t.next_up = t1.up; t.next_down = down

            # Remplir next_down du parent up si absent
            if t.up and t.up.next_down is None:
                t.up.next_down = down

            # Descendre d'un cran
            t, t1 = t.down, t1.down
            if t is None or t1 is None:
                break

    def _compute_branch_contributions(self, parent: Node, direction: str) -> Tuple[float, float, float]:
        """
        Calcule les contributions théoriques d'un futur noeud (up ou down) AVANT création.
        - parent.proba * p_(up|down)
        - + (proba de l'ancien up/down) * p_mid (chemin croisé)
        Retourne (contrib_directe, contrib_chemin_croisé, total).
        """
        if direction == "up":
            contrib_from_mid = parent.proba * self.p_up
            contrib_from_prev_dir_mid = (parent.up.proba * self.p_mid) if parent.up else 0.0
        else:
            contrib_from_mid = parent.proba * self.p_down
            contrib_from_prev_dir_mid = (parent.down.proba * self.p_mid) if parent.down else 0.0
        total = contrib_from_mid + contrib_from_prev_dir_mid
        return contrib_from_mid, contrib_from_prev_dir_mid, total

    def _maybe_prune_branch(self, parent: Node, next_mid: Node, direction: str, total_future_prob: float) -> bool:
        """
        Pruning uniquement aux bords : on ne prune que si le parent n'a pas de up (bord haut) ou pas de down (bord bas).
        """
        # Vérifie si pruning activé et masse trop faible
        if self.pruning and total_future_prob < self.epsilon:
            # Pruning uniquement si on est au bord
            is_bord = (direction == "up" and parent.up is None) or (direction == "down" and parent.down is None)
            if is_bord:
                next_mid.proba += parent.proba
                parent.next_mid = next_mid
                if direction == "up":
                    parent.next_up = None
                else:
                    parent.next_down = None
                return True
        return False

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
    
    def _create_next_trunc(self, node: Node, i: int) -> Node:
        """
        Étape ex-div:
        - On veut que le mid (colonne t+1) soit le plus proche de l'espérance = forward - D.
        - On génère plusieurs candidats autour de s_mid (par puissances de alpha).
        - On choisit le candidat le plus proche de l'espérance.
        - Puis on recalcule les probabilités en mode dividend=True.
        """
        esperance = compute_forward(node.S, self.market.r, self.delta_t) - self.market.dividend
        s_mid = node.S * np.exp(self.market.r * self.delta_t)

        # génère plusieurs candidats autour de s_mid
        candidates = []
        for k in range(-i-1, i):  # explore de alpha^-5 à alpha^5
            candidate = s_mid * (self.alpha ** k)
            candidates.append(candidate)
        # choisit le candidat le plus proche de l'espérance
        s_mid_best = min(candidates, key=lambda x: abs(x - esperance))


        next_mid = Node(S=s_mid_best, proba=0.0)
        next_mid.tree = self
        
        # rattachement central (utile pour backward)
        next_mid.trunc = next_mid; next_mid.prev_trunc = node
        next_mid.proba += node.proba * self.p_mid

        # Recalibrage des probas sur le bon mid ex-div
        self._compute_parameters(node.S, s_mid_best, dividend=True)

        return next_mid