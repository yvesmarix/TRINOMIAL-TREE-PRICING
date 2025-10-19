import numpy as np
from pricing.funcs import iter_column, compute_forward

class Node:

    def __init__(
        self,
        S,
        proba,
        up=None,
        down=None,
        next_up=None,
        next_mid=None,
        next_down=None,
        trunc=None,
        prev_trunc=None,
        option_value=None,
    ):
        self.S = S  # prix du sous-jacent au nœud
        self.proba = proba  # probabilité cumulée d'atteindre ce nœud depuis la racine
        self.up = up  # nœud "up" frère
        self.down = down  # nœud "down" frère
        self.next_up = next_up  # nœud "up" prochain
        self.next_mid = next_mid  # nœud "mid" prochain
        self.next_down = next_down  # nœud "down" prochain
        self.trunc=trunc # noeud central actuel
        self.prev_trunc=prev_trunc # noeud centrale précèdent
        self.option_value = option_value  # valeur de l'option au nœud
        self.tree = None # connaissance de l'arbre

    def create_mid(self, tree):
        """
        Crée le nœud 'mid' pour la prochaine colonne.
        """

        # pre-calculation
        proba_mid = self.proba * tree.p_mid
        s_mid = self._esperance(tree)
        
        # check si deja existant
        if self.down is not None and self.down.next_up is not None:
            self.next_mid = self.down.next_up; self.down.next_up.proba += proba_mid
        elif self.up is not None and self.up.next_down is not None:
            self.next_mid = self.up.next_down; self.up.next_down.proba += proba_mid
        else: # creation
            self.next_mid = Node(S=s_mid, proba=proba_mid)
            self.next_mid.tree = tree
            # rattachement central
            self._attach_trunc_links(self.next_mid)

    def prune_monomial(self, tree, dividend: bool):
        # cible = enfant le plus proche de l'espérance
        E = self._esperance(tree, is_dividend_step=dividend)

        # si on a déjà une colonne t+1 (créée par un voisin), choisir le plus proche
        cand = self._find_closest_node(E)
        if cand is not None:
            target = cand
        else:
            # aucun enfant existant
            target = Node(S=E, proba=0.0)
            target.tree = tree
            self._attach_trunc_links(target)

        # envoie toute la masse du nœud vers la cible
        target.proba += self.proba

        # on met les trois pointeurs vers le même enfant
        self.next_up = self.next_mid = self.next_down = target

    def _esperance(self, tree, is_dividend_step: bool = False) -> float:
        forward = self.S * np.exp(tree.market.r * tree.delta_t)
        return forward - tree.market.dividend if is_dividend_step else forward

    def _attach_trunc_links(self, child):
        if self.trunc is None:
            self.trunc = self
        child.trunc = child if child.trunc is None else child.trunc
        child.prev_trunc = self.trunc

    def next_column_nodes(self):
        """
        Retourne la liste des nœuds déjà créés de la colonne suivante (t+1),
        même si self.next_mid n'est pas encore câblé (on passe alors par le centre).
        """
        # trouve le centre de t+1
        center = None
        if self.next_mid is not None:
            center = self.next_mid.trunc
        elif self.down is not None and self.down.next_mid is not None:
            center = self.down.next_mid.trunc
        elif self.up is not None and self.up.next_mid is not None:
            center = self.up.next_mid.trunc

        if center is None:
            return []
        col = []
        # fait toute la colonne
        for x in iter_column(center):
            col.append(x)
        return col


    def _find_closest_node(self, target_price: float, above: bool | None = None):
        """
        Trouve le nœud le plus proche de target_price parmi les voisins directs
        et leurs descendants, pour potentiellement devenir next_mid.
        """
        candidates = self.next_column_nodes()
        # Si aucun candidat valide, retourner None
        if not candidates:
            return None
        # on exclu le mid lui même
        candidates = [n for n in candidates if n is not self.next_mid]
        # filtre directionnel
        if above is True:
            candidates = [n for n in candidates if n.S > self.next_mid.S]
        elif above is False:
            candidates = [n for n in candidates if n.S < self.next_mid.S]
        # dernier filtre
        if not candidates:
            return None
        candidate = min(candidates, key=lambda n: abs(n.S - target_price))

        return candidate

    def _ensure_neighbor(self, tree: object):
        """
        Vérifie les noeuds autour pour les utiliser comme frères sinon crée.
        """
        # partie down
        if self.next_down is None:
            cand = self._find_closest_node(self.next_mid.S / tree.alpha, above=False)
            if cand is not None:
                self.next_down = cand; self.next_down.proba += self.proba * tree.p_down
            else:
                self.create_down(tree)

        if self.next_down is not None: # s'assure de ne pas écraser des liaisons
            if self.next_mid.down is None or self.next_mid.down is self.next_down:
                self.next_mid.down = self.next_down
            if self.next_down.up is None or self.next_down.up is self.next_mid:
                self.next_down.up = self.next_mid

        # partie up
        if self.next_up is None:
            cand = self._find_closest_node(self.next_mid.S * tree.alpha, above=True)
            if cand is not None:
                self.next_up = cand; self.next_up.proba += self.proba * tree.p_up
            else:
                self.create_up(tree)

        if self.next_up is not None: # s'assure de ne pas écraser des liaisons
            if self.next_mid.up is None or self.next_mid.up is self.next_up:
                self.next_mid.up = self.next_up
            if self.next_up.down is None or self.next_up.down is self.next_mid:
                self.next_up.down = self.next_mid

    def _probas_valid(self, tree) -> bool:
        """Vérifie p in [0,1] et somme≈1 pour l'état courant du tree.
        Ce qui vérifie : somme pondérée des spot = esperance
        """
        p_down, p_up, p_mid = tree.p_down, tree.p_up, tree.p_mid
        if not (0.0 <= p_down <= 1.0 and 0.0 <= p_mid <= 1.0 and 0.0 <= p_up <= 1.0):
            return False
        return abs((p_down + p_mid + p_up) - 1.0) <= 1e-12

    def _shift_toward_expectation(self, esperance: float) -> bool:
        """
        Tente de déplacer next_mid d'un cran vers l'espérance.
        Retourne True si déplacement effectué, False sinon.
        """        
        mid = self.next_mid
        if mid is None:
            return False

        # proba à transférer avec les probas courantes (après _compute_parameters)
        proba_mid = self.proba * self.tree.p_mid
        # on choisi le sens de décalage
        if esperance > mid.S and mid.up is not None:
            new_mid = mid.up
        elif esperance < mid.S and mid.down is not None:
            new_mid = mid.down
        else:
            return False

        # déplace la masse cumulée vers le nouveau mid
        mid.proba -= proba_mid
        new_mid.proba += proba_mid

        self.next_mid = new_mid
        return True

    def _recenter_mid_until_valid(self, tree, i: int, esperance: float, dividend: bool):
        """
        Boucle : recalibre les probas avec le mid courant ;
        si invalide, décale le mid vers l'espérance et recommence.
        """
        for _ in range(i):
            # recalibrage autour du mid courant
            tree._compute_parameters(self.S, self.next_mid.S, dividend=dividend, validate=False)
            if self._probas_valid(tree):
                return
            moved = self._shift_toward_expectation(esperance)
            if not moved:
                break

    def create_mid_w_div(self, tree, i: int):
        """
        Crée le noeud du milieu avec comme objectif de le rattacher au noeud le plus proche
        de l'espérance. Au noeud centrale le spot est l'espérance (forward - dividende).
        Ensuite cherche des frères existants potentiels sinon crée les noeuds manquant.
        """
        esperance = self._esperance(tree, True)

        # choisi le noeud (sans toucher aux probas)
        cand = self._find_closest_node(esperance)
        self.next_mid = cand if cand is not None else Node(S=esperance, proba=0.0)
        self.next_mid.tree = tree
        self._attach_trunc_links(self.next_mid)

        # recentre jusqu’à probas valides
        self._recenter_mid_until_valid(tree, i, esperance, dividend=True)

        # crédite la masse correcte
        self.next_mid.proba += self.proba * tree.p_mid

        # crée/raccorde up/down
        self._ensure_neighbor(tree)


    def create_up(self, tree: object):
        """
        Crée le nœud 'up' pour la prochaine colonne.
        """
        # pre-calculation
        proba_up = self.proba * tree.p_up

        # ne pas recréer si déjà câblé
        if self.next_up is not None:
            self.next_up.proba += proba_up
            return
        
        # check si noeud deja existant
        if self.up is not None and self.up.next_mid is not None:
            self.next_up = self.up.next_mid; self.up.next_mid.proba += proba_up
        else:
            self.next_up = Node(S=self.next_mid.S * tree.alpha, proba=proba_up)
            self.next_up.tree = tree
            self.next_up.trunc = self.next_mid.trunc; self.next_up.prev_trunc = self.trunc

            # ajout des liens verticaux
            self.next_mid.up = self.next_up; self.next_up.down = self.next_mid

    def create_down(self, tree: object):
        """
        Crée le nœud 'down' pour la prochaine colonne.
        """
        # pre-calculation
        proba_down = self.proba * tree.p_down
        
        # ne pas recréer si déjà câblé
        if self.next_down is not None:
            self.next_down.proba += proba_down
            return

        # check si noeud deja existant
        if self.down is not None and self.down.next_mid is not None:
            self.next_down = self.down.next_mid; self.down.next_mid.proba += proba_down
        else:
            self.next_down = Node(S=self.next_mid.S / tree.alpha, proba=proba_down)
            self.next_down.tree = tree; self.next_down.trunc = self.next_mid.trunc
            self.next_down.prev_trunc = self.trunc
            # ajout des liens verticaux
            self.next_mid.down = self.next_down; self.next_down.up = self.next_mid
    
    def create_children(self, tree, i, dividend: bool = False):
        """
        Crée les enfants (up, mid, down) pour ce nœud.
        - Utilise les méthodes spécialisées pour chaque branche.
        """
        # crée le mid
        if not dividend: 
            self.create_mid(tree)
            # crée le up
            self.create_up(tree)
            # crée le down
            self.create_down(tree)
        else: 
            self.create_mid_w_div(tree, i)

    
    def price_recursive(self, option) -> float:
        """
        Calcule la valeur de l'option à ce nœud de façon récursive.
        Empêche la récursion infinie en mémorisant les valeurs déjà calculées.
        Utilise un ensemble 'visited' pour éviter les cycles.
        """
        # Si la valeur de l'option a déjà été calculée, on la retourne
        if self.option_value is not None:
            return self.option_value

        # payoffs aux feuilles
        if self.next_up is None and self.next_mid is None and self.next_down is None:
            self.option_value = option.payoff(self.S)
            return self.option_value
        
        # calcul récursif des enfants
        v_up = self.tree.p_up * (self.next_up.price_recursive(option) if self.next_up else 0.0)
        v_mid = self.tree.p_mid * (self.next_mid.price_recursive(option) if self.next_mid else 0.0)
        v_down = self.tree.p_down * (self.next_down.price_recursive(option) if self.next_down else 0.0)

        # actualisation
        discount = np.exp(-self.tree.market.r * self.tree.delta_t)
        continuation = discount * (v_up + v_mid + v_down)

        if option.option_class == "american":
            exercise = option.payoff(self.S)
            self.option_value = max(exercise, continuation)
        else:
            self.option_value = continuation
        return self.option_value
    
    def price_backward(self, option=None):
        """
        Backward induction explicite (sans récursion) avec factorisation.
        1. Trouve la dernière colonne via next_mid.
        2. Initialise les payoffs aux feuilles.
        3. Remonte colonne par colonne via prev_trunc.
        """
        if option is None:
            raise ValueError("Option requise pour le pricing backward.")

        # aller jusqu'à la dernière colonne (chaîne des next_mid)
        last_mid = self.tree.root
        while last_mid.next_mid:
            last_mid = last_mid.next_mid

        # initialisation des payoffs aux feuilles
        for leaf in iter_column(last_mid):
            leaf.option_value = option.payoff(leaf.S)

        p_up = self.tree.p_up
        p_mid = self.tree.p_mid
        p_down = self.tree.p_down
        discount = np.exp(-self.tree.market.r * self.tree.delta_t)
        is_american = (option.option_class == "american")
        payoff = option.payoff

        def compute_node(n):
            if n.option_value is not None:
                return
            vu = p_up * (n.next_up.option_value if n.next_up else 0.0)
            vm = p_mid * (n.next_mid.option_value if n.next_mid else 0.0)
            vd = p_down * (n.next_down.option_value if n.next_down else 0.0)
            cont = discount * (vu + vm + vd)
            if is_american:
                ex = payoff(n.S)
                n.option_value = ex if ex > cont else cont
            else:
                n.option_value = cont

        # remontée colonne par colonne
        mid = last_mid
        while mid.prev_trunc:
            prev_mid = mid.prev_trunc
            for n in iter_column(prev_mid):
                compute_node(n)
            mid = prev_mid

        return self.tree.root.option_value
