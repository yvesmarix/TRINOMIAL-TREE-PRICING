import numpy as np
import datetime as dt
from node import Node
from model import Model
from funcs import compute_forward, compute_variance, compute_probabilities


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

        return self.root.price_recursive(self.option)

    def _build_tree(self) -> Node:
        """
        On fait la méthode du remplissage verticale.
        On part du milieu et on va de haut en bas.
        """
        # racine
        root = Node(S=self.market.S0, proba=1.0)
        prev_mid = root

        # étape ex-div (ou None)
        div_step = self._compute_div_step()
        postdiv_set = False

        for i in range(self.N):
            step_left = self.N - (i + 1)
            has_div = getattr(self.market, "dividend", 0) > 0
            is_exdiv = (div_step is not None and (i + 1) == div_step and has_div)

            if is_exdiv:
                # seuil de faisabilité du mode "probas-only"
                forward = compute_forward(prev_mid.S, self.market.r, self.delta_t)
                Dmax = forward * (1.0 - 1.0 / self.alpha)

                if self.market.dividend >= Dmax - 1e-12:
                    # on shift la colonne de D
                    next_mid = self._create_next_mid(prev_mid)
                    self._extend_upper_part(next_mid, i, step_left)
                    self._extend_lower_part(next_mid, i, step_left)

                    # soustrait D
                    self._div_shifter(next_mid)

                    # lie les colonnes (next_*)
                    prev_mid = self._link_columns(prev_mid, next_mid)

                    # calibrage post-div (UNE fois) pour le reste des pas
                    if not postdiv_set:
                        self._compute_parameters(prev_mid.S, dividend=False)
                        postdiv_set = True
                    continue
                else:
                    # gestion du div avec les probas au pas ex-div 
                    self._compute_parameters(prev_mid.S, dividend=True)

            # construction standard avec les probas courantes
            next_mid = self._create_next_mid(prev_mid)
            self._extend_upper_part(next_mid, i, step_left)
            self._extend_lower_part(next_mid, i, step_left)
            prev_mid = self._link_columns(prev_mid, next_mid)

            # bascule post-div juste après l’ex-date (UNE fois)
            if is_exdiv and not postdiv_set:
                self._compute_parameters(prev_mid.S, dividend=False)
                postdiv_set = True

        root.tree = self
        return root

    def _compute_parameters(self, S: float, dividend: bool) -> None:
        # on calcule delta t
        self.delta_t = (self.option.maturity - self.pricing_date).days / self.N / 365

        # calcul de alpha
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.delta_t))

        # Probabilités
        forward = compute_forward(S, self.market.r, self.delta_t)
        if dividend:
            esperance = forward - self.market.dividend
        else:
            esperance=forward
        variance = compute_variance(
            S, self.market.r, self.delta_t, self.market.sigma
        )

        self.p_down, self.p_up, self.p_mid = compute_probabilities(
            esperance, forward, variance, self.alpha, dividend
        )

        # vérification des probabilités
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

    def _create_next_mid(self, prev_mid: Node) -> Node:
        """Crée le noeud central de la colonne t+1 à partir de prev_mid (colonne t)."""
        mid_price = prev_mid.S * np.exp(self.market.r * self.delta_t)
        next_mid = Node(S=mid_price, proba=prev_mid.proba * self.p_mid)
        next_mid.tree = self
        return next_mid

    def _extend_upper_part(self, next_mid: Node, i: int, step_left: int) -> None:
        """Construit la partie haute de la colonne t+1 à partir du milieu."""
        current = next_mid
        for _ in range(i + 1):
            s_up = current.S * self.alpha
            p_up = current.proba * self.p_up
            if self.pruning and self.__should_filter(s_up, step_left):
                break
            up = Node(S=s_up, proba=p_up)
            up.tree = self
            up.down = current
            current.up = up
            current = up

    def _extend_lower_part(self, next_mid: Node, i: int, step_left: int) -> None:
        """Construit la partie basse de la colonne t+1 à partir du milieu."""
        current = next_mid
        for _ in range(i + 1):
            s_down = current.S / self.alpha
            p_down = current.proba * self.p_down
            if self.pruning and self.__should_filter(s_down, step_left):
                break
            down = Node(S=s_down, proba=p_down)
            down.tree = self
            down.up = current
            current.down = down
            current = down


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
    
    def _div_shifter(self, next_mid: Node) -> None:
        # shift inline de toute la colonne : S <- max(S - D, 0)
        top = next_mid
        while top.up: top = top.up
        cur = top
        while cur:
            cur.S = max(cur.S - self.market.dividend, 0.0)
            cur = cur.down
