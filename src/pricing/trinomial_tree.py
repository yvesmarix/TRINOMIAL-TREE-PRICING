import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from typing import Literal, Optional, List, Tuple
from matplotlib.collections import LineCollection
from model import Model

from node import Node
from option import Option
from market import Market

import time
from funcs import *


class TrinomialTree(Model):
    """
    Arbre trinomial (construction verticale colonne par colonne).
    Gere pruning, dividendes et calculs de grecs.
    """

    def __init__(
        self,
        market: Market,
        N: int,
        pruning: bool = False,
        epsilon: Optional[float] = None,
        pricingDate: Optional[dt.datetime] = None,
    ) -> None:
        """Initialise le modele avec parametres de marche et profondeur N."""
        self.market = market
        self.N = N
        # decision de pruner (utile pour du debug)
        self.pruning = pruning
        self.epsilon = epsilon if epsilon is not None else 1e-7
        super().__init__(pricingDate or dt.datetime.today())

    # ------------------------------------------------------------------ #
    # Pricing
    # ------------------------------------------------------------------ #
    def price(
        self,
        option: Option,
        method: Literal["backward", "recursive"] = "backward",
        build_tree: bool = False,
        compute_greeks: bool = False,
        activate_timer: bool = False
    ) -> float:
        """
        Calcule le prix d'une option via arbre trinomial (backward ou recursif).
        Si build_tree=True, reconstruit l’arbre complet avant le pricing.
        """
        start = time.perf_counter()  # début du timer
        if build_tree or not hasattr(self, "root") or self.root is None:
            self.option = option
            self._compute_parameters(self.market.S0, dividend=False)
            self.root = self._build_tree(compute_greeks)

        price = self.root.price(option, method)
        elapsed = time.perf_counter() - start  # fin du timer
        if activate_timer: print(f"[Timer] Pricing exécuté en {elapsed:.4f} secondes.")   
        return price
    
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_tree(self, compute_greeks: bool = False) -> Node:
        """Construit l'arbre colonne par colonne et renvoie la racine."""
        # on instancie la racine
        root = Node(S=self.market.S0, proba=1.0)
        root.tree, root.trunc = self, root
        # si on veut delta gamma en un pricing on fait un noeud
        # au dessus/dessous decales de log(alpha)
        if compute_greeks:
            self._create_root_siblings_for_greeks(root)

        t = root # premiere colonne
        div_step = self._compute_div_step() # nombre de delta_t au div

        for i in range(self.N):
            dividend = (div_step == i + 1 and self.market.dividend > 0)
            for node in iter_column(t):
                if self._should_prune_node(node): # decision de pruner
                    node.prune_monomial(self, dividend)
                else:
                    node.create_children(self, i, dividend) # creation du triplet
            if dividend: # on revient aux anciennes probas
                self._compute_parameters(t.S)
            t = t.next_mid
        return root

    # ------------------------------------------------------------------ #
    # Grecs
    # ------------------------------------------------------------------ #
    def delta(self) -> float:
        """Δ via bump multiplicatif centre en log(S)."""
        h = self._get_bump()
        S0 = self.market.S0
        return (self.root.up.option_value - self.root.down.option_value) / (2 * S0 * h)

    def gamma(self) -> float:
        """Γ via derivees finies dans l’espace log(S)."""
        h, S0 = self._get_bump(), self.market.S0
        Vu, Vd, V0 = self.root.up.option_value, self.root.down.option_value, self.root.option_value
        g1, g2 = (Vu - Vd) / (2 * h), (Vu - 2 * V0 + Vd) / (h**2)
        return (g2 - g1) / (S0**2)

    def vega(self, option: Option, bump: float = 0.01) -> float:
        """Vega : derivee du prix par rapport à la volatilite sigma."""
        sigma0 = self.market.sigma
        self.market.sigma = sigma0 * (1 + bump)
        up = self.price(option, build_tree=True)
        self.market.sigma = sigma0 * (1 - bump)
        down = self.price(option, build_tree=True)
        self.market.sigma = sigma0
        return (up - down) / (2 * sigma0 * bump) / 100

    def vanna(self, option: Option, bump_sigma: float = 0.01, bump_S: float = 0.01) -> float:
        """Vanna : derivee croisee (∂²V / ∂S∂sigma)."""
        S0, sigma0 = self.market.S0, self.market.sigma

        # V(S+, sigma+)
        self.market.S0, self.market.sigma = S0 * (1 + bump_S), sigma0 * (1 + bump_sigma)
        up_up = self.price(option, build_tree=True)
        # V(S-, sigma-)
        self.market.S0, self.market.sigma = S0 * (1 - bump_S), sigma0 * (1 - bump_sigma)
        dn_dn = self.price(option, build_tree=True)
        # V(S+, sigma-)
        self.market.S0, self.market.sigma = S0 * (1 + bump_S), sigma0 * (1 - bump_sigma)
        up_dn = self.price(option, build_tree=True)
        # V(S-, sigma+)
        self.market.S0, self.market.sigma = S0 * (1 - bump_S), sigma0 * (1 + bump_sigma)
        dn_up = self.price(option, build_tree=True)

        # on retabli les parametres
        self.market.S0, self.market.sigma = S0, sigma0

        return (up_up + dn_dn - up_dn - dn_up) / (4 * S0 * sigma0 * bump_S * bump_sigma) / 100

    def rho(self, option: Option, bump: float = 0.0001) -> float:
        """Rho : derivee du prix par rapport au taux sans risque r."""
        r0 = self.market.r
        self.market.r = r0 + bump
        up = self.price(option, build_tree=True)
        self.market.r = r0 - bump
        down = self.price(option, build_tree=True)
        self.market.r = r0
        return (up - down) / (2 * bump) / 100

    def _get_bump(self) -> float:
        """Renvoie le pas log du bump (ln(alpha))."""
        if not getattr(self.root, "up", None) or not getattr(self.root, "down", None):
            raise RuntimeError("Activez compute_greeks dans price().")
        return np.log(self.alpha)

    def _create_root_siblings_for_greeks(self, root: Node) -> None:
        """Cree deux nœuds freres pour Δ et Γ.
        --> Utile pour ne faire qu'un seul pricing pour delta et gamma.
        """
        u, S = self.alpha, root.S
        up, down = Node(S * u, 1.0), Node(S / u, 1.0)
        for n in (up, down):
            n.tree, n.trunc, n.prev_trunc = self, root, None
        root.up, root.down = up, down
        up.down, down.up = root, root

    # ------------------------------------------------------------------ #
    # Parametres locaux
    # ------------------------------------------------------------------ #
    def _should_prune_node(self, node: Node) -> bool:
        """Retourne True si la proba du nœud < epsilon (pruning actif)."""
        return self.pruning and node.proba < self.epsilon

    def _compute_parameters(
        self,
        S: float,
        S1_mid: Optional[float] = None,
        dividend: bool = False,
        validate: bool = True,
    ) -> None:
        """Calibre les parametres alpha, p_up, p_mid, p_down."""
        self.delta_t = (self.option.maturity - self.pricing_date).days / self.N / 365
        self.alpha = np.exp(self.market.sigma * np.sqrt(3 * self.delta_t))
        # inputes probas
        fwd = compute_forward(S, self.market.r, self.delta_t)
        E = fwd - self.market.dividend if dividend else fwd
        var = compute_variance(S, self.market.r, self.delta_t, self.market.sigma)
        # calcule des probas
        self.p_down, self.p_up, self.p_mid = compute_probabilities(
            E, S1_mid if dividend else fwd, var, self.alpha, dividend
        )
        # check bornes
        if validate:
            probas_valid(self)

    def _compute_div_step(self) -> Optional[int]:
        """Renvoie l’etape correspondant à la date de dividende, ou None."""
        if not getattr(self.market, "dividend_date", None) or not self.market.dividend:
            return None
        years = (self.market.dividend_date - self.pricing_date).days / 365
        step = int(np.floor(years / self.delta_t + 1e-12))
        return max(1, min(step, self.N))

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def plot_tree(
        self,
        max_depth: Optional[int] = None,
        proba_min: float = 1e-9,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        percentile_clip: float = 0.0,
        edge_alpha: float = 0.35,
        linewidth: float = 0.4,
    ) -> None:
        """Trace l’arbre trinomial jusqu’à `max_depth`."""
        self._assert_tree_built()  # on s'assure d'abord qu'on a bien construit l'arbre
        depth = self.N if max_depth is None else max_depth  # profondeur cible (par defaut toute la hauteur)
        # on recupere les points et les arêtes dejà dedupliques
        xs, ys, sizes, edges, all_S = self._collect_nodes_and_edges(depth, proba_min)
        if not xs:
            # affichage inutile si on a tout filtre (probas trop petites)
            raise RuntimeError("aucun nœud à afficher (proba trop faible).")
        # bornes verticales automatiques (ou forcees si y_min/y_max fournis)
        y_min, y_max = self._compute_ylim(all_S, y_min, y_max, percentile_clip)
        # rendu final
        self._draw_graph(xs, ys, sizes, edges, y_min, y_max, edge_alpha, linewidth)


    def _assert_tree_built(self) -> None:
        """Verifie que l’arbre a ete construit avant affichage."""
        if not hasattr(self, "root") or self.root is None:
            # on evite de faire un joli plot… du vide
            raise RuntimeError("arbre non construit. utilisez price(build_tree=True).")


    def _collect_nodes_and_edges(
        self, depth: int, proba_min: float
    ) -> Tuple[List[float], List[float], List[float], List[Tuple[tuple, tuple]], List[float]]:
        """Parcours bfs pour recuperer positions, tailles et arêtes du graphe."""
        from collections import deque

        visited, seen = set(), set()      # pour eviter de repasser sur les mêmes nœuds/segments
        q = deque([(n, 0) for n in iter_column(self.root)])  # colonne 0 : racine + eventuels freres
        xs, ys, sizes, edges, all_S = [], [], [], [], []     # buffers de sortie

        def x_of(_node, lvl): 
            # on force x=0 sur la colonne 0 (racine et freres) pour un alignement propre
            return 0.0 if lvl == 0 else float(lvl)

        while q:
            n, lvl = q.popleft()
            if not n or lvl > depth or id(n) in visited:
                # on saute si on a depasse la profondeur, ou dejà vu
                continue
            visited.add(id(n))

            p = float(getattr(n, "proba", 0.0))
            S = getattr(n, "S", None)
            if S is None:
                # pas de spot = pas d’affichage
                continue
            all_S.append(S)

            if p >= proba_min:
                # on garde le nœud s'il porte assez de masse
                xs.append(x_of(n, lvl))
                ys.append(S)
                # taille proportionnelle à la proba (borne basse pour voir qqch)
                sizes.append(max(min(p, 1.0), 1e-16) * 1500)

                # on ajoute les segments vers les enfants (up/mid/down)
                for ch in (n.next_up, n.next_mid, n.next_down):
                    if ch and (id(n), id(ch)) not in seen:
                        edges.append([(x_of(n, lvl), S), (x_of(ch, lvl + 1), ch.S)])
                        seen.add((id(n), id(ch)))

            # on pousse les enfants dans la queue bfs (même si proba trop faible pour le display)
            for ch in (n.next_up, n.next_mid, n.next_down):
                if ch:
                    q.append((ch, lvl + 1))

        return xs, ys, sizes, edges, all_S


    def _compute_ylim(
        self,
        all_S: List[float],
        y_min: Optional[float],
        y_max: Optional[float],
        clip: float,
    ) -> Tuple[float, float]:
        """Definit les bornes y du trace automatiquement ou manuellement."""
        Svals = np.asarray(all_S)
        if 0.0 < clip < 0.5:
            # on clippe les extrêmes (optionnel) pour eviter des axes trop etires
            lo, hi = np.quantile(Svals, clip), np.quantile(Svals, 1 - clip)
        else:
            lo, hi = Svals.min(), Svals.max()

        # petite marge pour que les points ne touchent pas les bords
        span = max(1e-12, hi - lo)
        lo, hi = lo - 0.03 * span, hi + 0.03 * span

        # si l’utilisateur force y_min/y_max, on respecte
        return y_min or lo, y_max or hi


    def _draw_graph(
        self,
        xs: List[float],
        ys: List[float],
        sizes: List[float],
        edges: List[Tuple[tuple, tuple]],
        y_min: float,
        y_max: float,
        edge_alpha: float,
        linewidth: float,
    ) -> None:
        """Trace les liens et les nœuds (linecollection + scatter)."""
        fig, ax = plt.subplots(figsize=(14, 9))

        # les arêtes en LineCollection : beaucoup plus rapide que plein de plot()
        if edges:
            lc = LineCollection(edges, linewidths=linewidth, alpha=edge_alpha, rasterized=True)
            ax.add_collection(lc)

        # les nœuds : un scatter rasterized pour rester fluide quand il y en a beaucoup
        ax.scatter(xs, ys, s=sizes, alpha=0.7, rasterized=True, linewidths=0)

        # titres/axes : on reste sobre
        ax.set(title="arbre trinomial", xlabel="etapes (t)", ylabel="sous-jacent S")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, ls="--", alpha=0.35)

        fig.tight_layout()
        plt.show()
