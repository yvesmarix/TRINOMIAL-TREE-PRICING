import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_tree(tree):
    """
    Génère un graphique horizontal de l'arbre trinomial avec :
    - Taille des nœuds proportionnelle à la probabilité du nœud.
    - Structure claire avec les étapes sur l'axe vertical.
    
    :param tree: Instance de TrinomialTree
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    nodes = []  # Liste des positions des nœuds
    sizes = []  # Liste des tailles des nœuds
    edges = []  # Liste des connexions entre les nœuds

    def collect_node_data(node, level, x_pos):
        """
        Collecte les données des nœuds et des connexions.
        """
        if node is None:
            return
        
        # Ajouter le nœud
        nodes.append((level, x_pos))  # Position horizontale basée sur le niveau
        sizes.append(max(node.proba * 2000, 10))  # Taille minimale pour éviter les nœuds invisibles

        # Ajouter les connexions
        if node.next_up:
            edges.append([(level, x_pos), (level + 1, x_pos + 1)])
            collect_node_data(node.next_up, level + 1, x_pos + 1)
        if node.next_mid:
            edges.append([(level, x_pos), (level + 1, x_pos)])
            collect_node_data(node.next_mid, level + 1, x_pos)
        if node.next_down:
            edges.append([(level, x_pos), (level + 1, x_pos - 1)])
            collect_node_data(node.next_down, level + 1, x_pos - 1)

    # Collecter les données à partir de la racine
    collect_node_data(tree.root, level=0, x_pos=0)

    # Tracer les connexions avec LineCollection
    line_collection = LineCollection(edges, colors="gray", linewidths=0.5, alpha=0.7)
    ax.add_collection(line_collection)

    # Tracer les nœuds
    nodes_x, nodes_y = zip(*nodes)
    ax.scatter(nodes_x, nodes_y, s=sizes, color="blue", alpha=0.6)

    # Configurer les axes
    ax.set_title("Arbre Trinomial (horizontal)", fontsize=14)
    ax.set_xlabel("Étapes", fontsize=12)
    ax.set_ylabel("Position horizontale", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()