class Node:
    def __init__(self, S, proba, up=None, down=None, next_up=None, next_mid=None, next_down=None):
        self.S = S      # Prix du sous-jacent au nœud
        self.proba = proba  # Probabilité d'atteindre ce nœud
        self.up = up    # Nœud "up" frère
        self.down = down# Nœud "down" frère
        self.next_up = next_up    # Nœud "up" prochain
        self.next_mid = next_mid  # Nœud "mid" prochain
        self.next_down = next_down# Nœud "down" prochain
        # il faudra rajouter option value ici pour le pricing
        # il faut donner l'objet option à la racine de l’arbre
        # on doit pouvoir faire tree.root.price(option)
        # pour le pricing si next mid est None alors on est à la fin de l’arbre et on doit pricer

        # la valeur intrinsèque ou payoff doit être une methode de la classe option 
        # et on doit lui donner node.S pour pricer