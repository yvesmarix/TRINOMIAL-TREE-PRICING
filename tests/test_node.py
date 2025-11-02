from __future__ import annotations

import math
import random
import types
from typing import List, Tuple, Callable

import numpy as np
import pytest

from pricing.node import Node


# ---------------------------------------------------------------------------
# Helpers typés
# ---------------------------------------------------------------------------
def _mock_tree_discount(r: float, delta_t: float) -> object:
    market = types.SimpleNamespace(r=r)
    return types.SimpleNamespace(market=market, delta_t=delta_t)


def _mock_tree_probabilities(
    p_up: float,
    p_mid: float,
    p_down: float,
) -> object:
    return types.SimpleNamespace(
        p_up=p_up,
        p_mid=p_mid,
        p_down=p_down,
    )


def _make_euro_option(strike: float = 100.0) -> object:
    class Option:
        option_class: str = "european"

        def __init__(self, K: float) -> None:
            self.K = K

        def payoff(self, S: float) -> float:
            return max(S - self.K, 0.0)

    return Option(strike)


def _make_american_option(strike: float = 100.0) -> object:
    class Option:
        option_class: str = "american"

        def __init__(self, K: float) -> None:
            self.K = K

        def payoff(self, S: float) -> float:
            return max(S - self.K, 0.0)

    return Option(strike)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_node_initialization() -> None:
    n: Node = Node(S=100.0, proba=0.5)
    assert n.S == 100.0
    assert n.proba == 0.5
    assert n.up is None and n.down is None
    assert n.next_up is None and n.next_mid is None and n.next_down is None
    assert n.option_value is None


def test_node_is_leaf() -> None:
    n: Node = Node(S=100.0, proba=1.0)
    assert n._is_leaf() is True
    n.next_up = Node(S=110.0, proba=0.3)
    assert n._is_leaf() is False


@pytest.mark.parametrize(
    "r,delta_t",
    [
        (
            random.uniform(0.01, 0.1),
            random.uniform(1 / 365, 1 / 12),
        )
        for _ in range(5)
    ],
)
def test_node_discount(r: float, delta_t: float) -> None:
    n: Node = Node(S=100.0, proba=1.0)
    n.tree = _mock_tree_discount(r, delta_t)  # type: ignore[assignment]
    discount: float = n._discount()
    assert discount == pytest.approx(np.exp(-r * delta_t))


@pytest.mark.parametrize(
    "p_up,p_mid,p_down,option_values",
    [(0.3, 0.4, 0.3, [10.0, 5.0, 0.0]) for _ in range(5)],
)
def test_node_continuation_from_children(
    p_up: float,
    p_mid: float,
    p_down: float,
    option_values: List[float],
) -> None:
    n: Node = Node(S=100.0, proba=1.0)
    n.tree = _mock_tree_probabilities(p_up, p_mid, p_down)  # type: ignore[assignment]

    n.next_up = Node(S=110.0, proba=0.0, option_value=option_values[0])
    n.next_mid = Node(S=100.0, proba=0.0, option_value=option_values[1])
    n.next_down = Node(S=90.0, proba=0.0, option_value=option_values[2])

    n._discount = lambda: 1.0  # type: ignore[assignment]
    cont: float = n._continuation_from_children()
    expected: float = (
        p_up * option_values[0]
        + p_mid * option_values[1]
        + p_down * option_values[2]
    )
    assert cont == pytest.approx(expected)


@pytest.mark.parametrize(
    "S,K,continuation,option_class,expected",
    [
        (100.0, 100.0, 2.0, "american", 2.0),
        (110.0, 100.0, 2.0, "american", 10.0),
        (100.0, 100.0, 2.0, "european", 2.0),
    ],
)
def test_node_apply_exercise_rule(
    S: float,
    K: float,
    continuation: float,
    option_class: str,
    expected: float,
) -> None:
    n: Node = Node(S=S, proba=1.0)
    option = (
        _make_american_option(K) if option_class == "american" else _make_euro_option(K)
    )
    res: float = n._apply_exercise_rule(option, continuation)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "p_up,p_mid,p_down,option_values",
    [(0.3, 0.4, 0.3, [10.0, 5.0, 0.0]) for _ in range(5)],
)
def test_node_price_recursive(
    p_up: float,
    p_mid: float,
    p_down: float,
    option_values: List[float],
) -> None:
    n: Node = Node(S=100.0, proba=1.0)
    n.next_up = Node(S=110.0, proba=p_up, option_value=option_values[0])
    n.next_mid = Node(S=100.0, proba=p_mid, option_value=option_values[1])
    n.next_down = Node(S=90.0, proba=p_down, option_value=option_values[2])

    class MockTree:
        def __init__(self) -> None:
            self.p_up = p_up
            self.p_mid = p_mid
            self.p_down = p_down
            self.market = types.SimpleNamespace(r=0.05)
            self.delta_t = 1 / 365

    n.tree = MockTree()  # type: ignore[assignment]
    option = _make_euro_option(100.0)
    n.option_value = None
    price: float = n.price_recursive(option)
    assert price > 0.0
