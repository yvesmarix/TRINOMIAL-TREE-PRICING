import numpy as np
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from trinomial_tree import TrinomialTree

from dataclasses import dataclass
from pricing.funcs import iter_column, probas_valid

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
        """evaluation recursive des sous-arbres (formule de valorisation)."""
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
    def create_mid(self, tree: Optional["TrinomialTree"]) -> None:
        """Cree le nœud central (mid) pour la colonne suivante."""
        proba_mid = self.proba * tree.p_mid
        s_mid = self._esperance(tree)

        if self.down and self.down.next_up: # logique fixe sans decalage
            # cumule des probas pour le pruning
            self.next_mid = self.down.next_up; self.down.next_up.proba += proba_mid
            
        elif self.up and self.up.next_down: # on va chercher les noeuds existants
            # cumule des probas pour le pruning
            self.next_mid = self.up.next_down; self.up.next_down.proba += proba_mid
        
        else: # sinon on cree
            self.next_mid = Node(S=s_mid, proba=proba_mid, tree=tree)
            self._attach_trunc_links(self.next_mid) # rattachement au trunc

    def create_mid_w_div(self, tree: Optional["TrinomialTree"], i: int) -> None:
        """Cree le mid en tenant compte d’un dividende à l’etape courante."""
        esp = self._esperance(tree, True)
        cand = self._find_closest_node(esp) # recupere le meilleur candidat

        self.next_mid = cand or Node(S=esp, proba=0.0, tree=tree)
        self._attach_trunc_links(self.next_mid) # rattachement au trunc

        # decalage tant que c'est pas borne
        self._recenter_mid_until_valid(tree, i, esp, dividend=True)
        self.next_mid.proba += self.proba * tree.p_mid # cumule des probas pour le pruning
        self._ensure_neighbor(tree) # verifie s'il existe des noeuds susceptibles d'être les freres

    def _create_neighbor(self, tree: Optional["TrinomialTree"], kind: str):
        """
        Methode qui crée le noeud up ou down.
        """
        factor = tree.alpha if kind=="up" else 1/tree.alpha
        p = self.proba * (tree.p_up if kind=="up" else tree.p_down)
        next_attr = "next_up" if kind=="up" else "next_down"
        sibling_attr = "up" if kind=="up" else "down"

        # déjà existant
        if getattr(self, next_attr):
            setattr(getattr(self, next_attr), "proba", getattr(self, next_attr).proba + p)
            return

        # décalage fixe
        sib = getattr(self, sibling_attr)
        if sib and sib.next_mid:
            setattr(self, next_attr, sib.next_mid); sib.next_mid.proba += p
            return

        # sinon on crée
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
        if not dividend:
            self.create_mid(tree); self._create_neighbor(tree, 'up'); self._create_neighbor(tree, 'down')
        else:
            self.create_mid_w_div(tree, i)

    def prune_monomial(self, tree: Optional["TrinomialTree"], dividend: bool) -> None:
        """Prune les branches faibles (redirection vers le mid le plus proche)."""
        E = self._esperance(tree, is_dividend_step=dividend)
        
        # candidat au mid le plus proche
        cand = self._find_closest_node(E)
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

    def _next_column_nodes(self) -> list["Node"]:
        """Retourne tous les noeuds de la colonne suivante (t+1)."""
        # on cherche un centre valide pour t+1 (via mid sinon via voisins)
        candidates = [self.next_mid,
                    self.down.next_mid if self.down else None,
                    self.up.next_mid if self.up else None]
        for c in candidates:
            if c: return list(iter_column(c.trunc))
        return []

    def _find_closest_node(self, target: float, above: Optional[bool] = None) -> Optional["Node"]:
        """Trouve le noeud le plus proche du prix cible (dans la colonne suivante)."""
        # candidates = tous les noeuds de t+1 sauf notre mid courant
        cand = [n for n in self._next_column_nodes() if n is not self.next_mid]
        # filtre directionnel optionnel (utile quand on sait de quel cote chercher)
        if above is True:
            cand = [n for n in cand if n.S > self.next_mid.S]   # on force au-dessus
        elif above is False:
            cand = [n for n in cand if n.S < self.next_mid.S]   # on force en-dessous
        # on renvoie le plus proche si existe
        return min(cand, key=lambda n: abs(n.S - target)) if cand else None

    def _ensure_neighbor(self, tree: Optional["TrinomialTree"]) -> None:
        """Relie ou cree les voisins up/down autour du mid."""
        # partie down: on reutilise un voisin proche si possible sinon on cree
        if not self.next_down:
            cand = self._find_closest_node(self.next_mid.S / tree.alpha, above=False)
            if cand:
                self.next_down, cand.proba = cand, cand.proba + self.proba * tree.p_down
            else:
                self._create_neighbor(tree, "down")
        # partie up: idem
        if not self.next_up:
            cand = self._find_closest_node(self.next_mid.S * tree.alpha, above=True)
            if cand:
                self.next_up, cand.proba = cand, cand.proba + self.proba * tree.p_up
            else:
                self._create_neighbor(tree, "up")

    def _shift_toward_expectation(self, E: float) -> bool:
        """Deplace next_mid d un cran vers l esperance."""
        mid = self.next_mid
        if not mid:
            return False  # rien a faire si pas de mid
        proba_mid = self.proba * self.tree.p_mid  # masse a deplacer
        # choix du sens: up si E>mid.S, sinon down si E<mid.S
        if E > mid.S and mid.up:
            new_mid = mid.up
        elif E < mid.S and mid.down:
            new_mid = mid.down
        else:
            return False  # bloque aux bords
        # transfert de masse et mise a jour du pointeur
        mid.proba -= proba_mid; new_mid.proba += proba_mid; self.next_mid = new_mid
        return True

    def _recenter_mid_until_valid(self, tree: Optional["TrinomialTree"], i: int, E: float, dividend: bool) -> None:
        """Recentre le mid jusqu a obtenir des probabilites valides."""
        # on tente i iterations max: recalibre puis, si invalide, on bouge vers E
        for _ in range(i):
            tree._compute_parameters(self.S, self.next_mid.S, dividend=dividend, validate=False)
            if probas_valid(tree):
                return  # ok, on arrete
            if not self._shift_toward_expectation(E):
                break  # on ne peut plus bouger

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

    def _reset_option_values_column(self, mid) -> None:
        """Reinitialise les valeurs d option d une colonne."""
        # pratique si on relance un pricing sur une colonne deja visitee
        for n in iter_column(mid):
            n.option_value = None
