import pytest
import random
from pricing.node import Node

import numpy as np

def test_node_initialization():
    """Test de l'initialisation d'un nœud."""
    n = Node(S=100.0, proba=0.5)
    assert n.S == 100.0
    assert n.proba == 0.5
    assert n.up is None
    assert n.down is None
    assert n.next_up is None
    assert n.next_mid is None
    assert n.next_down is None
    assert n.option_value is None

def test_node_is_leaf():
    """Test pour vérifier si un nœud est une feuille."""
    n = Node(S=100.0, proba=1.0)
    assert n._is_leaf() is True

    n.next_up = Node(S=110.0, proba=0.3)
    assert n._is_leaf() is False

@pytest.mark.parametrize(
    "r, delta_t",
    [(random.uniform(0.01, 0.1), random.uniform(1 / 365, 1 / 12)) for _ in range(5)]
)
def test_node_discount(r, delta_t, monkeypatch):
    """Test du facteur d'actualisation."""
    n = Node(S=100.0, proba=1.0)
    class MockTree:
        def __init__(self, r, delta_t):
            self.market = type("Market", (), {"r": r})
            self.delta_t = delta_t
    n.tree = MockTree(r, delta_t)

    discount = n._discount()
    assert discount == pytest.approx(np.exp(-r * delta_t))

@pytest.mark.parametrize(
    "p_up, p_mid, p_down, option_values",
    [(0.3, 0.4, 0.3, [10.0, 5.0, 0.0]) for _ in range(5)]
)
def test_node_continuation_from_children(p_up, p_mid, p_down, option_values, monkeypatch):
    """Test de l'espérance actualisée des valeurs des enfants."""
    n = Node(S=100.0, proba=1.0)
    class MockTree:
        def __init__(self, p_up, p_mid, p_down):
            self.p_up = p_up
            self.p_mid = p_mid
            self.p_down = p_down
    n.tree = MockTree(p_up, p_mid, p_down)

    n.next_up = Node(S=110.0, proba=0.0, option_value=option_values[0])
    n.next_mid = Node(S=100.0, proba=0.0, option_value=option_values[1])
    n.next_down = Node(S=90.0, proba=0.0, option_value=option_values[2])

    monkeypatch.setattr(n, "_discount", lambda: 1.0)
    continuation = n._continuation_from_children()
    assert continuation == pytest.approx(p_up * option_values[0] + p_mid * option_values[1] + p_down * option_values[2])

@pytest.mark.parametrize(
    "S, K, continuation, option_class, expected",
    [
        (100.0, 100.0, 2.0, "american", 2.0),
        (110.0, 100.0, 2.0, "american", 10.0),
        (100.0, 100.0, 2.0, "european", 2.0),
    ]
)
def test_node_apply_exercise_rule(S, K, continuation, option_class, expected):
    """Test de la règle d'exercice pour les options européennes et américaines."""
    n = Node(S=S, proba=1.0)
    class Option:
        def __init__(self, option_class, K):
            self.option_class = option_class
            self.K = K
        def payoff(self, S): return max(S - self.K, 0.0)

    option = Option(option_class, K)
    assert n._apply_exercise_rule(option, continuation) == pytest.approx(expected)

@pytest.mark.parametrize(
    "p_up, p_mid, p_down, option_values",
    [(0.3, 0.4, 0.3, [10.0, 5.0, 0.0]) for _ in range(5)]
)
def test_node_price_recursive(p_up, p_mid, p_down, option_values):
    """Test du pricing récursif."""
    n = Node(S=100.0, proba=1.0)
    n.next_up = Node(S=110.0, proba=p_up, option_value=option_values[0])
    n.next_mid = Node(S=100.0, proba=p_mid, option_value=option_values[1])
    n.next_down = Node(S=90.0, proba=p_down, option_value=option_values[2])

    class Option:
        option_class = "european"
        def payoff(self, S): return max(S - 100, 0.0)

    class MockTree:
        def __init__(self, p_up, p_mid, p_down):
            self.p_up = p_up
            self.p_mid = p_mid
            self.p_down = p_down
            self.market = type("Market", (), {"r": 0.05})
            self.delta_t = 1 / 365

    n.tree = MockTree(p_up, p_mid, p_down)

    n.option_value = None
    price = n.price_recursive(Option())
    assert price > 0