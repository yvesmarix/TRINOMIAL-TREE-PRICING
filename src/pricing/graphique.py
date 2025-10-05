import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_tree(
    root_node,
    size_scale: float = 4000.0,
    prob_min: float = 0.0,
    draw_edges: bool = True,
    max_columns: int | None = None
):
    """
    Trace un arbre trinomial où la taille de chaque nœud est proportionnelle à sa probabilité.
    Hypothèses sur les attributs du nœud: S, proba, up, down, next_up, next_mid, next_down.
    """
    x_positions_by_step: list[float] = []
    y_underlying_prices: list[float] = []
    node_probabilities:   list[float] = []
    edge_segments:        list[list[tuple[float, float]]] = []

    current_column_mid = root_node
    step_index = 0

    # Collecte efficace colonne par colonne
    while current_column_mid is not None and (max_columns is None or step_index < max_columns):
        # Monter au sommet de la colonne
        top_node = current_column_mid
        while getattr(top_node, "up", None) is not None:
            top_node = top_node.up

        # Redescendre la colonne en ajoutant points et segments
        node = top_node
        while node is not None:
            p = float(getattr(node, "proba", 0.0) or 0.0)
            if p >= prob_min:
                s = float(getattr(node, "S", 0.0) or 0.0)
                x_positions_by_step.append(step_index)
                y_underlying_prices.append(s)
                node_probabilities.append(p)

                if draw_edges:
                    for link_name in ("next_down", "next_mid", "next_up"):
                        child = getattr(node, link_name, None)
                        if child is not None:
                            p_child = float(getattr(child, "proba", 0.0) or 0.0)
                            if p_child >= prob_min:
                                s_child = float(getattr(child, "S", 0.0) or 0.0)
                                edge_segments.append([(step_index, s), (step_index + 1, s_child)])

            node = getattr(node, "down", None)

        current_column_mid = getattr(current_column_mid, "next_mid", None)
        step_index += 1

    # Mise à l'échelle des tailles de marqueurs
    probabilities_array = np.asarray(node_probabilities)
    max_probability = probabilities_array.max() if probabilities_array.size else 1.0
    marker_sizes = size_scale * (probabilities_array / (max_probability if max_probability > 0 else 1.0))

    # Tracé
    fig, ax = plt.subplots(figsize=(10, 5))
    if draw_edges and edge_segments:
        ax.add_collection(LineCollection(edge_segments, linewidths=0.4, alpha=0.25))
    ax.scatter(
        x_positions_by_step, y_underlying_prices,
        s=marker_sizes, alpha=0.9, linewidths=0
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Underlying S")
    ax.margins(x=0.02, y=0.05)
    plt.tight_layout()
    return fig
