def plot_tree(
    root_node,
    size_scale: float = 4000.0,
    prob_min: float = 0.0,
    draw_edges: bool = True,
    max_columns: int | None = None
):
    """
    Trace un arbre trinomial où la taille de chaque nœud est proportionnelle à sa probabilité.
    Fonction plus robuste : évite les boucles infinies, gère les nodes manquants et
    n'essaie de tracer les arêtes que si l'enfant appartient bien à la colonne suivante.
    """
    import math

    # Si l'objet tree fournit sa propre méthode de collecte de colonnes, on la réutilise (plus fiable).
    cols = None
    try:
        if hasattr(root_node, "tree") and hasattr(root_node.tree, "_collect_columns"):
            cols = root_node.tree._collect_columns(root_node)
    except Exception:
        cols = None

    # Fallback : collecte sûre colonne par colonne
    if cols is None:
        cols = []
        visited = set()
        cur = root_node
        step = 0
        while cur is not None and (max_columns is None or step < max_columns):
            # remonter au sommet de la colonne
            top = cur
            # protégeons contre boucles mal formées
            seen_climb = set()
            while getattr(top, "up", None) is not None and id(top) not in seen_climb:
                seen_climb.add(id(top))
                top = top.up

            # descendre la colonne en collectant les noeuds (unique par id)
            col = []
            n = top
            while n is not None:
                nid = id(n)
                if nid not in visited:
                    col.append(n)
                    visited.add(nid)
                n = getattr(n, "down", None)
                # protection contre boucle locale
                if n is not None and id(n) in visited and getattr(n, "down", None) is n:
                    break

            if not col:
                # pas de noeud trouvés -> on arrête la collecte
                break

            cols.append(col)

            next_mid = getattr(cur, "next_mid", None)
            if next_mid is cur:
                break
            cur = next_mid
            step += 1

    # Construire mapping node_id -> column index pour filtrer les arêtes
    node_to_col = {}
    for xi, col in enumerate(cols):
        for node in col:
            node_to_col[id(node)] = xi

    # Collections pour le tracé
    x_positions_by_step = []
    y_underlying_prices = []
    node_probabilities = []
    edge_segments = []

    # Parcourir colonnes et remplir les listes (en triant par S pour cohérence visuelle)
    for xi, col in enumerate(cols):
        # Trier la colonne par S décroissant (top -> bottom)
        try:
            col_sorted = sorted(col, key=lambda n: float(getattr(n, "S", 0.0) or 0.0), reverse=True)
        except Exception:
            col_sorted = col

        for node in col_sorted:
            try:
                p = float(getattr(node, "proba", 0.0) or 0.0)
                s = float(getattr(node, "S", 0.0) or 0.0)
            except Exception:
                # ignorer nodes malformés
                continue

            if not (math.isfinite(p) and math.isfinite(s)):
                continue
            if p < prob_min:
                continue

            x_positions_by_step.append(xi)
            y_underlying_prices.append(s)
            node_probabilities.append(p)

            if draw_edges:
                for link_name in ("next_down", "next_mid", "next_up"):
                    child = getattr(node, link_name, None)
                    if child is None:
                        continue
                    # Ne tracer l'arête que si l'enfant est bien dans la colonne suivante
                    cid = id(child)
                    child_col = node_to_col.get(cid, None)
                    if child_col is None or child_col != xi + 1:
                        continue
                    try:
                        s_child = float(getattr(child, "S", 0.0) or 0.0)
                    except Exception:
                        continue
                    if not math.isfinite(s_child):
                        continue
                    edge_segments.append([(xi, s), (xi + 1, s_child)])

    # Mise à l'échelle des tailles de marqueurs (protection contre division par 0)
    import numpy as np
    probabilities_array = np.asarray(node_probabilities)
    if probabilities_array.size:
        max_probability = float(np.nanmax(probabilities_array))
        if not np.isfinite(max_probability) or max_probability <= 0:
            max_probability = 1.0
    else:
        max_probability = 1.0
    marker_sizes = size_scale * (probabilities_array / max_probability) if probabilities_array.size else []

    # Tracé
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(10, 5))
    if draw_edges and edge_segments:
        ax.add_collection(LineCollection(edge_segments, linewidths=0.4, alpha=0.25))
    if len(x_positions_by_step):
        ax.scatter(
            x_positions_by_step, y_underlying_prices,
            s=marker_sizes if len(marker_sizes) else 16, alpha=0.9, linewidths=0
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Underlying S")
    ax.margins(x=0.02, y=0.05)
    plt.tight_layout()
    plt.show()
