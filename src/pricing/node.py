from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import numpy as np

from pricing.funcs import iter_column, compute_probabilities, compute_forward

if TYPE_CHECKING:
    from pricing.trinomial_tree import TrinomialTree
    from pricing.option import Option


@dataclass
class Node:
    """Nœud d’un arbre trinomial (spot, proba, liens et valeur d’option)."""

    S: float
    proba: float
    up: Optional["Node"] = None
    down: Optional["Node"] = None
    next_up: Optional["Node"] = None
    next_mid: Optional["Node"] = None
    next_down: Optional["Node"] = None
    trunc: Optional["Node"] = None
    prev_trunc: Optional["Node"] = None
    option_value: Optional[float] = None
    tree: Optional["TrinomialTree"] = None

    # paramètres LOCAUX (uniquement utilisés sur le pas de dividende,
    # quand on a "snappé" le mid et qu'on veut conserver l'espérance)
    local_alpha: Optional[float] = None
    local_p_up: Optional[float] = None
    local_p_mid: Optional[float] = None
    local_p_down: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Pricing
    # ------------------------------------------------------------------ #
    def price(self, option: "Option", method: str = "backward") -> float:
        """Calcule la valeur au nœud selon la méthode choisie."""
        if method == "backward": return self.price_backward(option)
        if method == "recursive": return self.price_recursive(option)
        raise ValueError("method doit être 'backward' ou 'recursive'.")

    def price_recursive(self, option: "Option") -> float:
        """Évaluation récursive des sous-arbres."""
        if self.option_value is not None: return self.option_value  # déjà calculé
        if self._is_leaf():  # feuille : payoff direct
            self.option_value = option.payoff(self.S); return self.option_value

        # partie récursive : on propage d'abord sur les enfants
        v_up  = self.next_up.price_recursive(option)   if self.next_up  else 0.0
        v_mid = self.next_mid.price_recursive(option)  if self.next_mid else 0.0
        v_dn  = self.next_down.price_recursive(option) if self.next_down else 0.0

        # on stocke les valeurs sur les enfants (évite les re-call)
        if self.next_up:   self.next_up.option_value   = v_up
        if self.next_mid:  self.next_mid.option_value  = v_mid
        if self.next_down: self.next_down.option_value = v_dn

        # valeur de continuation (espérance actualisée) + exercice
        cont = self._continuation_from_children()
        self.option_value = self._apply_exercise_rule(option, cont)
        return self.option_value

    def price_backward(self, option: "Option") -> float:
        """Évaluation backward (colonne par colonne)."""
        if option is None: raise ValueError("Option requise pour le pricing backward.")

        # aller au bout de la colonne centrale (trunc de fin)
        last_mid = self.tree.root
        while last_mid.next_mid: last_mid = last_mid.next_mid

        # payoff aux feuilles
        for leaf in iter_column(last_mid): leaf.option_value = option.payoff(leaf.S)

        # remontée colonne par colonne
        mid = last_mid
        while mid.prev_trunc:
            prev_mid = mid.prev_trunc
            for n in iter_column(prev_mid):
                cont = n._continuation_from_children()
                n.option_value = n._apply_exercise_rule(option, cont)
            mid = prev_mid
        return self.tree.root.option_value  # valeur à la racine

    # ------------------------------------------------------------------ #
    # Construction des enfants
    # ------------------------------------------------------------------ #
    def create_mid(self, tree: Optional["TrinomialTree"], dividend: bool = False) -> None:
        """
        Crée le nœud central (mid) pour la colonne suivante.
        Cas standard : réutilise ou crée au niveau de l'espérance, cumule la proba.
        Cas dividende : on "snap" vers le plus proche puis on recalcule des probas LOCALES.
        """
        esp_target = self._esperance(tree, is_dividend_step=dividend)  # espérance cible
        p_mid = self.proba * tree.p_mid  # masse de proba à transférer
        self.next_mid = (
            self.down.next_up  if self.down and self.down.next_up  else
            self.up.next_down  if self.up   and self.up.next_down  else
            Node(S=esp_target, proba=(0.0 if dividend else p_mid), tree=tree)
        )
        # rattachement au tronc / cumul
        if self.next_mid.trunc is None: self._attach_trunc_links(self.next_mid)
        elif not dividend: self.next_mid.proba += p_mid
        if not dividend: return  # cas normal terminé

        # --- cas dividende ---
        self._handle_dividend_case(tree, esp_target, p_mid)

    def _handle_dividend_case(self, tree: Optional["TrinomialTree"], esp_target: float, p_mid: float) -> None:
        """
        Gère le cas particulier du pas de dividende :
        - décale le mid vers l’espérance cible,
        - recalcule les probas LOCALES pour conserver l’espérance du parent,
        - stocke ces paramètres sur le nœud central.
        """
        # se rapprocher de l'espérance
        self._shift_mid_toward_expectation(esp_target, tree)

        # parametres locaux
        alpha_loc = np.exp(tree.market.sigma * np.sqrt(3 * tree.delta_t))  # alpha local
        fwd_from_snapped = self.next_mid.S * np.exp(tree.market.r * tree.delta_t)
        var_from_snapped = self.next_mid.S**2 * np.exp(2 * tree.market.r * tree.delta_t) * (np.exp(tree.market.sigma**2 * tree.delta_t) - 1)

        # proba LOCALES pour conserver l'espérance du parent
        p_d, p_u, p_m_loc = compute_probabilities(
            esperance=esp_target,
            forward=fwd_from_snapped,
            variance=var_from_snapped,
            alpha=alpha_loc,
            dividend=True,
        )

        # stockage sur le mid
        self._assign_local_probs(self.next_mid, alpha_loc, p_u, p_m_loc, p_d)
        self.next_mid.proba += p_mid  # cumule la proba du parent
        self._ensure_neighbor(tree)   # voisins cohérents

    def _shift_mid_toward_expectation(self, esp: float, tree: Optional["TrinomialTree"]) -> None:
        """Décale next_mid vers le nœud le plus proche de l'espérance, limité à i déplacements."""

        while True:
            # candidats up/down (créés si absents) autour du mid courant
            up = self.next_mid.up or Node(S=self.next_mid.S * tree.alpha, proba=0.0, tree=tree)

            if self.next_mid.up is None:  # rattachements
                up.trunc, up.prev_trunc, up.down, self.next_mid.up = self.next_mid.trunc, self.trunc, self.next_mid, up
            dn = self.next_mid.down or Node(S=self.next_mid.S / tree.alpha, proba=0.0, tree=tree)
            if self.next_mid.down is None:
                dn.trunc, dn.prev_trunc, dn.up, self.next_mid.down = self.next_mid.trunc, self.trunc, self.next_mid, dn

            # erreurs
            cur_err = abs(self.next_mid.S - esp); up_err = abs(up.S - esp); dn_err = abs(dn.S - esp)

            # se déplacer vers le plus proche ; s'arrêter si rien n'améliore
            if up_err < cur_err and up_err <= dn_err: self.next_mid = up; continue
            if dn_err < cur_err and dn_err <  up_err: self.next_mid = dn; continue
            break

    def _create_neighbor(self, tree: Optional["TrinomialTree"], kind: str) -> None:
        """
        Crée le nœud up ou down autour de self.next_mid,
        en héritant des paramètres LOCAUX du mid s'il en a.
        """
        factor = tree.alpha if kind == "up" else 1 / tree.alpha
        p = self.proba * (tree.p_up if kind == "up" else tree.p_down)
        next_attr = "next_up" if kind == "up" else "next_down"
        sibling_attr = "up" if kind == "up" else "down"

        # déjà existant
        if getattr(self, next_attr): getattr(self, next_attr).proba += p; return

        # réutilisation du mid du sibling
        sib = getattr(self, sibling_attr)
        if sib and sib.next_mid: setattr(self, next_attr, sib.next_mid); sib.next_mid.proba += p; return

        # sinon on crée
        node = Node(S=self.next_mid.S * factor, proba=p, tree=tree)
        node.trunc, node.prev_trunc = self.next_mid.trunc, self.trunc

        # héritage des paramètres LOCAUX
        self._copy_local_probs(self.next_mid, node)

        # connexions verticales
        if kind == "up":   self.next_mid.up,   node.down = node, self.next_mid
        else:              self.next_mid.down, node.up   = node, self.next_mid
        setattr(self, next_attr, node)

    def create_children(self, tree: Optional["TrinomialTree"], dividend: bool = False) -> None:
        """Crée les enfants up/mid/down du nœud."""
        self.create_mid(tree, dividend); self._create_neighbor(tree, "up"); self._create_neighbor(tree, "down")

    def prune_monomial(self, tree: Optional["TrinomialTree"], dividend: bool) -> None:
        """Prune les branches faibles (redirection vers le mid le plus proche)."""
        E = self._esperance(tree, is_dividend_step=dividend)  # espérance (ajustée si dividende)

        # candidat existant
        cand = (self.down.next_up if self.down and self.down.next_up else
                self.up.next_down  if self.up   and self.up.next_down  else None)
        target = cand or Node(S=E, proba=0.0); target.tree = tree

        if not cand:  # rattachement + lien vertical si besoin
            self._attach_trunc_links(target)
            if self.up is None and self.down is not None: target.down = self.next_mid; self.down.next_mid.up = target
            elif self.down is None and self.up is not None: target.up = self.next_mid; self.up.next_mid.down = target

        # propage aussi les paramètres LOCAUX si on est sur le pas dividende
        if dividend: self._copy_local_probs(self, target)

        target.proba += self.proba  # cumule les probas
        self.next_mid = target      # redirection

    # ------------------------------------------------------------------ #
    # Outils internes
    # ------------------------------------------------------------------ #
    def _assign_local_probs(self, node: "Node", alpha: float, p_up: float, p_mid: float, p_down: float) -> None:
        """Stocke les probas locales sur un nœud."""
        node.local_alpha  = alpha
        node.local_p_up   = p_up
        node.local_p_mid  = p_mid
        node.local_p_down = p_down

    def _copy_local_probs(self, src: "Node", dst: "Node") -> None:
        """Copie les probas locales si présentes."""
        if src.local_p_up is None and src.local_p_mid is None and src.local_p_down is None: return
        dst.local_alpha  = src.local_alpha
        dst.local_p_up   = src.local_p_up
        dst.local_p_mid  = src.local_p_mid
        dst.local_p_down = src.local_p_down

    def _esperance(self, tree: Optional["TrinomialTree"], is_dividend_step: bool = False) -> float:
        """Renvoie le forward espéré (ajusté du dividende si besoin)."""
        fwd = compute_forward(self.S, tree.market.r, tree.delta_t)
        return fwd - tree.market.dividend if is_dividend_step else fwd

    def _attach_trunc_links(self, child: "Node") -> None:
        """Relie le noeud enfant à la chaîne centrale (trunc)."""
        self.trunc = self.trunc or self    # si rien, le tronc c'est lui
        child.trunc = child.trunc or child # l'enfant pointe sur son propre tronc si besoin
        child.prev_trunc = self.trunc      # lien vers le tronc précédent

    def _ensure_neighbor(self, tree: Optional["TrinomialTree"]) -> None:
        """Relie ou crée les voisins up/down autour du mid courant."""
        if not self.next_down: self._create_neighbor(tree, "down")
        if not self.next_up:   self._create_neighbor(tree, "up")

    # ------------------------------------------------------------------ #
    # Fonctions de base du pricing
    # ------------------------------------------------------------------ #
    def _is_leaf(self) -> bool:
        """Renvoie True si le noeud est une feuille (aucun enfant)."""
        return not (self.next_up or self.next_mid or self.next_down)

    def _discount(self) -> float:
        """Facteur d’actualisation du pas dt."""
        return np.exp(-self.tree.market.r * self.tree.delta_t)

    def _continuation_from_children(self) -> float:
        """Espérance actualisée des valeurs des enfants."""

        # priorité aux paramètres locaux (colonne du dividende)
        pu = self.local_p_up  if self.local_p_up  is not None else self.tree.p_up
        pm = self.local_p_mid if self.local_p_mid is not None else self.tree.p_mid
        pd = self.local_p_down if self.local_p_down is not None else self.tree.p_down

        # somme p*V enfant (0 si enfant absent)
        vu = pu * (self.next_up.option_value   if self.next_up   and self.next_up.option_value   is not None else 0.0)
        vm = pm * (self.next_mid.option_value  if self.next_mid  and self.next_mid.option_value  is not None else 0.0)
        vd = pd * (self.next_down.option_value if self.next_down and self.next_down.option_value is not None else 0.0)

        return self._discount() * (vu + vm + vd)  # actualisation

    def _apply_exercise_rule(self, option: "Option", cont: float) -> float:
        """Applique la règle d’exercice (Euro ou Américain)."""
        if option.option_class == "american":
            return max(option.payoff(self.S), cont)
        return cont

