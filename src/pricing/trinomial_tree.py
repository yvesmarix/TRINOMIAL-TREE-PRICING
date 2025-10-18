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
            dividend = (div_step == i + 1 and self.market.dividend > 0)

            # Crée les enfants pour chaque nœud de la colonne courante
            for node, is_bord in iter_column(t, mode="building"):
                node.create_children(self, pruning=self.pruning, is_bord=is_bord, dividend=dividend)

            # retour aux probas normales
            if dividend:
                self._compute_parameters(t.S)

            # Avance à la colonne suivante
            t = t.next_mid

        root.tree = self
        return root

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
        """BFS dédupliqué → positions, tailles, segments et liste des S pour les bornes."""
        from collections import deque
        from typing import List, Tuple

        visited, seen_edges = set(), set()
        queue = deque([(self.root, 0)])

        xs: List[int] = []
        ys: List[float] = []
        sizes: List[float] = []
        edges: List[Tuple[tuple, tuple]] = []
        all_S: List[float] = []

        def push_edge(parent_id, lvl, S0, child):
            if child is None: return
            eid = (parent_id, id(child))
            if eid not in seen_edges:
                edges.append([(lvl, S0), (lvl + 1, getattr(child, "S", S0))])
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
                xs.append(lvl); ys.append(S)
                sizes.append(max(min(p, 1.0), 1e-16) * 1500.0)

            up, mid, dn = getattr(node, "next_up", None), getattr(node, "next_mid", None), getattr(node, "next_down", None)

            # on pousse les arêtes seulement une fois par parent (dédup)
            if p >= proba_min:
                push_edge(nid, lvl, S, up)
                push_edge(nid, lvl, S, mid)
                push_edge(nid, lvl, S, dn)

            # parcourir toujours (un enfant faible peut mener à un nœud fort)
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
