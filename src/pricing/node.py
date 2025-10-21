from math import trunc
import numpy as np
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from trinomial_tree import TrinomialTree

from dataclasses import dataclass
from pricing.funcs import iter_column

@dataclass
class Node:
    """Nœud d’un arbre trinomial (spot, proba, liens et valeur d’option)."""

    S: float
    proba: float
    up: Optional["Node"]=None
    down: Optional["Node"]=None
    next_up: Optional["Node"]=None
    next_mid: Optional["Node"]=None
    next_down: Optional["Node"]=None
    trunc: Optional["Node"]=None
    prev_trunc: Optional["Node"]=None
    option_value: Optional[float]=None
    tree: Optional["TrinomialTree"]=None

    # ------------------------------------------------------------------ #
    # Pricing
    # ------------------------------------------------------------------ #
    def price(self, option, method: str = "backward") -> float:
        """Calcule la valeur au nœud selon la methode choisie."""
        if method == "backward":
            return self.price_backward(option)
        if method == "recursive":
            return self.price_recursive(option)
        raise ValueError("method doit être 'backward' ou 'recursive'.")

    def price_recursive(self, option) -> float:
        """evaluation recursive des sous-arbres."""
        if self.option_value is not None: # evite de recalculer pour rien
            return self.option_value
        if self._is_leaf(): # payoff aux feuilles
            self.option_value = option.payoff(self.S)
            return self.option_value
        
        # partie recursive
        v_up = self.next_up.price_recursive(option) if self.next_up else 0.0
        v_mid = self.next_mid.price_recursive(option) if self.next_mid else 0.0
        v_dn = self.next_down.price_recursive(option) if self.next_down else 0.0

        # attribution des valeurs
        if self.next_up: self.next_up.option_value = v_up
        if self.next_mid: self.next_mid.option_value = v_mid
        if self.next_down: self.next_down.option_value = v_dn

        # application du type d'exercice
        cont = self._continuation_from_children()
        self.option_value = self._apply_exercise_rule(option, cont)
        return self.option_value

    def price_backward(self, option) -> float:
        """evaluation backward (colonne par colonne)."""
        if option is None:
            raise ValueError("Option requise pour le pricing backward.")
        last_mid = self.tree.root

        while last_mid.next_mid: # direction la fin de l'arbre
            last_mid = last_mid.next_mid

        for leaf in iter_column(last_mid): # payoff aux feuilles
            leaf.option_value = option.payoff(leaf.S)

        mid = last_mid # on se place au trunc du fond
        while mid.prev_trunc:
            prev_mid = mid.prev_trunc
            for n in iter_column(prev_mid): # on traverse l'arbre en backward
                cont = n._continuation_from_children()
                n.option_value = n._apply_exercise_rule(option, cont)
            mid = prev_mid
        return self.tree.root.option_value

    # ------------------------------------------------------------------ #
    # Construction des enfants
    # ------------------------------------------------------------------ #
    
    def create_mid(self, tree: Optional["TrinomialTree"], dividend: bool = False) -> None:
        """Cree le nœud central (mid) pour la colonne suivante,  
        en gerant à la fois le cas standard et celui avec dividende (decalage vers l’esperance)."""
        # esperance (ajustee si dividende) et masse de proba mid
        esp, p_mid = self._esperance(tree, dividend), self.proba * tree.p_mid

        # point de depart : reutilise un node existant ou cree à l'esperance
        self.next_mid = (
            self.down.next_up if self.down and self.down.next_up else
            self.up.next_down if self.up and self.up.next_down else
            Node(S=esp, proba=(0.0 if dividend else p_mid), tree=tree)
        )
        if self.next_mid.trunc is None:
            self._attach_trunc_links(self.next_mid)
        elif not dividend:
            self.next_mid.proba += p_mid  # cumul proba si reutilise (cas sans dividende)

        # si dividende : decale vers le bas tant que plus proche de l'esperance
        if dividend:
            while True:
                dn = self.next_mid.down or Node(S=self.next_mid.S / tree.alpha, proba=0.0, tree=tree)
                if self.next_mid.down is None:
                    dn.trunc, dn.prev_trunc, dn.up, self.next_mid.down = self.next_mid.trunc, self.trunc, self.next_mid, dn
                if abs(dn.S - esp) < abs(self.next_mid.S - esp): self.next_mid = dn
                else: break
            # recalcule des paramètres
            tree._compute_parameters(self.S, self.next_mid.S, dividend=True, validate=True)
            self.next_mid.proba += p_mid # cumule des probas
            self._ensure_neighbor(tree)  # voisins up/down au pas dividende

    def _create_neighbor(self, tree: Optional["TrinomialTree"], kind: str):
        """
        Methode qui cree le noeud up ou down.
        """
        factor = tree.alpha if kind=="up" else 1/tree.alpha
        p = self.proba * (tree.p_up if kind=="up" else tree.p_down)
        next_attr = "next_up" if kind=="up" else "next_down"
        sibling_attr = "up" if kind=="up" else "down"

        # dejà existant
        if getattr(self, next_attr):
            setattr(getattr(self, next_attr), "proba", getattr(self, next_attr).proba + p)
            return

        # decalage fixe
        sib = getattr(self, sibling_attr)
        if sib and sib.next_mid:
            setattr(self, next_attr, sib.next_mid); sib.next_mid.proba += p
            return

        # sinon on cree
        node = Node(S=self.next_mid.S * factor, proba=p, tree=tree)
        node.trunc, node.prev_trunc = self.next_mid.trunc, self.trunc
        # connexions verticales
        if kind=="up":
            self.next_mid.up, node.down = node, self.next_mid
        else:
            self.next_mid.down, node.up = node, self.next_mid
        setattr(self, next_attr, node)

    def create_children(self, tree: Optional["TrinomialTree"], i: int, dividend: bool = False) -> None:
        """Cree les enfants up/mid/down du nœud."""
        self.create_mid(tree, dividend)
        self._create_neighbor(tree, 'up')
        self._create_neighbor(tree, 'down')

    def prune_monomial(self, tree: Optional["TrinomialTree"], dividend: bool) -> None:
        """Prune les branches faibles (redirection vers le mid le plus proche)."""
        E = self._esperance(tree, is_dividend_step=dividend)

        # candidat au mid le plus proche
        cand = (self.down.next_up if self.down and self.down.next_up else 
            self.up.next_down if self.up and self.up.next_down else None)
        target = cand or Node(S=E, proba=0.0)
        target.tree = tree # le noeud connait son arbre

        if not cand: self._attach_trunc_links(target) # rattachement au trunc
        target.proba += self.proba # cumule des probas
        self.next_up = self.next_mid = self.next_down = target

    # ------------------------------------------------------------------ #
    # Outils internes
    # ------------------------------------------------------------------ #
    def _esperance(self, tree: Optional["TrinomialTree"], is_dividend_step: bool = False) -> float:
        """Renvoie le forward espere (ajuste du dividende)."""
        fwd = self.S * np.exp(tree.market.r * tree.delta_t)  # forward simple
        return fwd - tree.market.dividend if is_dividend_step else fwd

    def _attach_trunc_links(self, child: "Node") -> None:
        """Relie le noeud enfant a la chaine centrale (trunc)."""
        self.trunc = self.trunc or self  # si rien, le tronc c'est lui
        child.trunc = child.trunc or child  # l'enfant pointe sur son propre tronc si besoin
        child.prev_trunc = self.trunc  # lien vers le tronc precedent

    def _ensure_neighbor(self, tree: Optional["TrinomialTree"]) -> None:
        # relie ou crée les voisins up/down autour du mid
        if not self.next_down: self._create_neighbor(tree, "down")
        if not self.next_up:   self._create_neighbor(tree, "up")

    # ------------------------------------------------------------------ #
    # Fonctions de base du pricing
    # ------------------------------------------------------------------ #
    def _is_leaf(self) -> bool:
        """Renvoie True si le noeud est une feuille (aucun enfant)."""
        return not (self.next_up or self.next_mid or self.next_down)

    def _discount(self) -> float:
        """Facteur d actualisation du pas dt."""
        return np.exp(-self.tree.market.r * self.tree.delta_t)

    def _continuation_from_children(self) -> float:
        """Esperance actualisee des valeurs des enfants."""
        # on recupere les proba locales
        pu, pm, pd = self.tree.p_up, self.tree.p_mid, self.tree.p_down
        # somme p*V enfant (0 si enfant absent)
        vu = pu * (self.next_up.option_value if self.next_up else 0.0)
        vm = pm * (self.next_mid.option_value if self.next_mid else 0.0)
        vd = pd * (self.next_down.option_value if self.next_down else 0.0)
        # on actualise la somme
        return self._discount() * (vu + vm + vd)

    def _apply_exercise_rule(self, option, cont: float) -> float:
        """Applique la regle d exercice (Euro ou Americain)."""
        # americain: max(payoff, continuation) ; sinon continuation
        return max(option.payoff(self.S), cont) if option.option_class == "american" else cont