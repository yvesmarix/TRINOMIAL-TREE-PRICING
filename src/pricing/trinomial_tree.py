import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import deque
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

    def price(
        self,
        build_tree: bool = False,
        compute_greeks: bool = False,
        bump_eps: float = 1e-3,
        bump_mode: str = "relative",
    ):
        """
        Calcule le prix (backward). Optionnellement, prépare deux frères au-dessus/en-dessous
        de la racine pour obtenir Δ et Γ en un seul backward.
        """
        if build_tree or not hasattr(self, "root") or self.root is None:
            self._compute_parameters(self.market.S0, dividend=False)
            self.root = self._build_tree(
                compute_greeks=compute_greeks,
                bump_eps=bump_eps,
                bump_mode=bump_mode,
            )

        v0 = self.root.price_backward(self.option)
        if not compute_greeks:
            return v0

        h = self._bump_size(bump_eps, bump_mode)
        greeks = self._extract_greeks_from_root(v0, h)
        return v0, greeks
    
    def _build_tree(
        self,
        compute_greeks: bool = False,
        bump_eps: float = 1e-3,
        bump_mode: str = "relative",
    ) -> Node:
        # racine
        root = Node(S=self.market.S0, proba=1.0); root.tree = self; root.trunc = root

        # grecs: on ajoute 2 frères sur la colonne 0
        if compute_greeks:
            self._create_root_siblings_for_greeks(root, bump_eps, bump_mode)

        # construction colonne par colonne
        t = root
        div_step = self._compute_div_step()

        for i in range(self.N):
            dividend = (div_step == i + 1 and self.market.dividend > 0)
            for node in iter_column(t):
                # si compute_greeks -> on force la création (évite pruning asymétrique)
                if (not compute_greeks) and self._should_prune_node(node):
                    node.prune_monomial(self, dividend=dividend)
                else:
                    node.create_children(self, i, dividend=dividend)
            if dividend:
                self._compute_parameters(t.S) # retour aux probas classiques
            t = t.next_mid

        return root

    def _bump_size(self, bump_eps: float, bump_mode: str) -> float:
        """Retourne h (taille du bump) selon le mode."""
        return (bump_eps * self.market.S0) if bump_mode == "relative" else bump_eps

    def _create_root_siblings_for_greeks(
        self, root: "Node", bump_eps: float, bump_mode: str
    ) -> None:
        """
        Crée deux nœuds FRÈRES (up/down) autour de la racine, dans la même colonne.
        Ce ne sont PAS des enfants (pas de next_*).
        """
        h = self._bump_size(bump_eps, bump_mode)
        sup = root.S + h
        sdn = root.S - h

        up0   = Node(S=sup, proba=1.0)
        down0 = Node(S=sdn, proba=1.0)

        up0.tree = self; down0.tree = self
        up0.trunc = root; down0.trunc = root
        up0.prev_trunc = None; down0.prev_trunc = None

        # liens verticaux (même colonne 0)
        root.up = up0; up0.down = root
        root.down = down0; down0.up = root

    def _extract_greeks_from_root(self, v0: float, h: float) -> dict:
        """
        Lit les valeurs déjà fixées par price_backward() sur root.up / root.down.
        Calcule Delta et Gamma centrés.
        """
        up_node = getattr(self.root, "up", None)
        dn_node = getattr(self.root, "down", None)
        if up_node is None or dn_node is None:
            return {"delta": None, "gamma": None}

        v_up = up_node.option_value
        v_dn = dn_node.option_value
        if v_up is None or v_dn is None:
            return {"delta": None, "gamma": None}

        delta = (v_up - v_dn) / (2.0 * h)
        gamma = (v_up - 2.0 * v0 + v_dn) / (h ** 2)
        return {"delta": float(delta), "gamma": float(gamma)}

    def _should_prune_node(self, node) -> bool:
        # full-monomial si la masse arrivée sur ce nœud est trop faible
        return self.pruning and (node.proba < self.epsilon)

    def _compute_parameters(self, S: float, S1_mid: float = None, dividend: bool = False, validate: bool = True) -> None:
        """
        Calibrage local.
        Si validate=False, on n'asserte pas (utile pendant le recentrage du mid).
        """
        self.delta_t = (self.option.maturity - self.pricing_date).days / self.N / 365
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.delta_t))

        forward  = compute_forward(S, self.market.r, self.delta_t)
        esperance = forward - self.market.dividend if dividend else forward
        variance = compute_variance(S, self.market.r, self.delta_t, self.market.sigma)

        self.p_down, self.p_up, self.p_mid = compute_probabilities(
            esperance, S1_mid if dividend else forward, variance, self.alpha, dividend
        )
        if validate:
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

    def plot_tree(
        self,
        max_depth=None,
        proba_min: float = 1e-9,
        y_min: float | None = None,
        y_max: float | None = None,
        percentile_clip: float = 0.0,
        edge_alpha: float = 0.35,
        linewidth: float = 0.4,
    ):
        """Tracé rapide : y = S, liens + nœuds, dédup, rasterize."""
        self._assert_tree_built()
        depth = self.N if max_depth is None else max_depth

        xs, ys, sizes, edges, all_S = self._collect_nodes_and_edges(
            depth=depth, proba_min=proba_min
        )
        if not xs:
            raise RuntimeError("Aucun nœud à afficher (proba_min trop élevée ?).")

        y_min, y_max = self._compute_ylim(all_S, y_min, y_max, percentile_clip)
        self._draw_graph(xs, ys, sizes, edges, y_min, y_max, edge_alpha, linewidth)

    def _assert_tree_built(self) -> None:
        if not hasattr(self, "root") or self.root is None:
            raise RuntimeError("Arbre non construit. Appelle price(build_tree=True).")

    def _collect_nodes_and_edges(self, depth: int, proba_min: float):
        """BFS dédupliqué → positions, tailles, segments et liste des S pour les bornes.
        Aligne tous les nœuds de la colonne 0 sur x=0.
        """
        from collections import deque
        from typing import List, Tuple

        visited, seen_edges = set(), set()
        queue = deque([(n, 0) for n in iter_column(self.root)])

        xs: List[float] = []
        ys: List[float] = []
        sizes: List[float] = []
        edges: List[Tuple[tuple, tuple]] = []
        all_S: List[float] = []

        def x_of(_node, lvl: int) -> float:
            # -> tous les nœuds de la colonne 0 (racine et ses frères) à x=0
            return 0.0 if lvl == 0 else float(lvl)

        def push_edge(parent, lvl: int, S0: float, child):
            if child is None:
                return
            eid = (id(parent), id(child))
            if eid not in seen_edges:
                px = x_of(parent, lvl)
                cx = x_of(child,  lvl + 1)
                edges.append([(px, S0), (cx, getattr(child, "S", S0))])
                seen_edges.add(eid)

        while queue:
            node, lvl = queue.popleft()
            if node is None or lvl > depth:
                continue
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)

            p = float(getattr(node, "proba", 0.0) or 0.0)
            S = getattr(node, "S", None)
            if S is None:
                continue
            all_S.append(S)

            if p >= proba_min:
                xs.append(x_of(node, lvl))        # x=0 pour t=0
                ys.append(S)
                sizes.append(max(min(p, 1.0), 1e-16) * 1500.0)

            up  = getattr(node, "next_up", None)
            mid = getattr(node, "next_mid", None)
            dn  = getattr(node, "next_down", None)

            if p >= proba_min:
                push_edge(node, lvl, S, up)
                push_edge(node, lvl, S, mid)
                push_edge(node, lvl, S, dn)

            if up:  queue.append((up,  lvl + 1))
            if mid: queue.append((mid, lvl + 1))
            if dn:  queue.append((dn,  lvl + 1))

        return xs, ys, sizes, edges, all_S


    def _compute_ylim(self, all_S, y_min, y_max, percentile_clip: float):
        """Bornes Y explicites ou automatiques (avec clip optionnel)."""
        import numpy as np
        if y_min is not None and y_max is not None:
            return y_min, y_max

        Svals = np.asarray(all_S, float)
        if 0.0 < percentile_clip < 0.5:
            lo = float(np.quantile(Svals, percentile_clip))
            hi = float(np.quantile(Svals, 1 - percentile_clip))
        else:
            lo, hi = float(Svals.min()), float(Svals.max())

        span = max(1e-12, hi - lo)
        lo -= 0.03 * span
        hi += 0.03 * span

        return (y_min if y_min is not None else lo,
                y_max if y_max is not None else hi)

    def _draw_graph(self, xs, ys, sizes, edges, y_min, y_max, edge_alpha, linewidth):
        """Rendu matplotlib performant : LineCollection + scatter rasterized."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig, ax = plt.subplots(figsize=(14, 9))

        if edges:
            lc = LineCollection(edges, linewidths=linewidth, alpha=edge_alpha)
            lc.set_rasterized(True)
            ax.add_collection(lc)

        ax.scatter(xs, ys, s=sizes, alpha=0.7, rasterized=True, linewidths=0)

        ax.set_title("Arbre trinomial", fontsize=14)
        ax.set_xlabel("Étapes (t)")
        ax.set_ylabel("Sous-jacent S")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        plt.show()
