from __future__ import annotations

import datetime as dt
import random
from typing import Tuple

import pytest

from pricing.trinomial_tree import TrinomialTree
from pricing import Market, Option


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_market(
    S0: float,
    r: float,
    sigma: float,
    dividend: float = 0.0,
    dividend_date: dt.datetime | None = None,
) -> Market:
    return Market(
        S0=S0,
        r=r,
        sigma=sigma,
        dividend=dividend,
        dividend_date=dividend_date,
    )


def _make_option_call(
    K: float,
    maturity: dt.datetime,
    option_class: str = "european",
) -> Option:
    return Option(
        K=K,
        option_type="call",
        maturity=maturity,
        option_class=option_class,
    )


def _make_tree(
    market: Market,
    N: int = 50,
    pruning: bool = False,
    epsilon: float = 1e-7,
    pricing_date: dt.datetime | None = None,
) -> TrinomialTree:
    return TrinomialTree(
        market=market,
        N=N,
        pruning=pruning,
        epsilon=epsilon,
        pricing_date=pricing_date or dt.datetime(2025, 1, 1),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "S0,r,sigma,dividend,N,epsilon",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(0, 10),
            random.randint(10, 100),
            1e-6,
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_initialization(
    S0: float,
    r: float,
    sigma: float,
    dividend: float,
    N: int,
    epsilon: float,
) -> None:
    market: Market = _make_market(S0, r, sigma, dividend)
    pricing_date: dt.datetime = dt.datetime(2025, 1, 1)
    tree: TrinomialTree = _make_tree(
        market=market,
        N=N,
        pruning=True,
        epsilon=epsilon,
        pricing_date=pricing_date,
    )
    assert tree.market is market
    assert tree.N == N
    assert tree.pruning is True
    assert tree.epsilon == epsilon
    assert tree.pricing_date == pricing_date


@pytest.mark.parametrize(
    "S0,r,sigma,dividend,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(0, 10),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_compute_parameters(
    S0: float,
    r: float,
    sigma: float,
    dividend: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(
        S0,
        r,
        sigma,
        dividend,
        dividend_date=dt.datetime(2025, 6, 1),
    )
    tree: TrinomialTree = _make_tree(
        market=market,
        N=5,
        pruning=False,
        epsilon=1e-7,
        pricing_date=dt.datetime(2025, 1, 1),
    )
    option: Option = _make_option_call(K, maturity)
    tree.option = option
    tree._compute_parameters(S=market.S0, validate=True)

    assert 0.0 <= tree.p_down <= 1.0
    assert 0.0 <= tree.p_up <= 1.0
    assert 0.0 <= tree.p_mid <= 1.0
    assert tree.p_down + tree.p_up + tree.p_mid == pytest.approx(1.0)
    assert tree.alpha > 1.0
    assert tree.delta_t > 0.0


@pytest.mark.parametrize(
    "S0,r,sigma,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_price(
    S0: float,
    r: float,
    sigma: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(S0, r, sigma, 0.0)
    option: Option = _make_option_call(K, maturity)
    tree: TrinomialTree = _make_tree(market, N=50)
    price: float = tree.price(option, method="backward", build_tree=True)
    assert price > 0.0


@pytest.mark.parametrize(
    "S0,r,sigma,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_delta(
    S0: float,
    r: float,
    sigma: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(S0, r, sigma, 0.0)
    option: Option = _make_option_call(K, maturity)
    tree: TrinomialTree = _make_tree(market, N=50)
    tree.price(option, compute_greeks=True)
    delta: float = tree.delta()
    assert isinstance(delta, float)


@pytest.mark.parametrize(
    "S0,r,sigma,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_gamma(
    S0: float,
    r: float,
    sigma: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(S0, r, sigma, 0.0)
    option: Option = _make_option_call(K, maturity)
    tree: TrinomialTree = _make_tree(market, N=50)
    tree.price(option, compute_greeks=True)
    gamma: float = tree.gamma()
    assert isinstance(gamma, float)


@pytest.mark.parametrize(
    "S0,r,sigma,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_vega(
    S0: float,
    r: float,
    sigma: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(S0, r, sigma, 0.0)
    option: Option = _make_option_call(K, maturity)
    tree: TrinomialTree = _make_tree(market, N=50)
    tree.price(option, compute_greeks=True)
    vega: float = tree.vega(option)
    assert isinstance(vega, float)


@pytest.mark.parametrize(
    "S0,r,sigma,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_vanna(
    S0: float,
    r: float,
    sigma: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(S0, r, sigma, 0.0)
    option: Option = _make_option_call(K, maturity)
    tree: TrinomialTree = _make_tree(market, N=50)
    tree.price(option, compute_greeks=True)
    vanna: float = tree.vanna(option)
    assert isinstance(vanna, float)


@pytest.mark.parametrize(
    "S0,r,sigma,K,maturity",
    [
        (
            random.uniform(80, 120),
            random.uniform(0.01, 0.05),
            random.uniform(0.1, 0.3),
            random.uniform(90, 110),
            dt.datetime(2025, 7, 1),
        )
        for _ in range(5)
    ],
)
def test_trinomial_tree_rho(
    S0: float,
    r: float,
    sigma: float,
    K: float,
    maturity: dt.datetime,
) -> None:
    market: Market = _make_market(S0, r, sigma, 0.0)
    option: Option = _make_option_call(K, maturity)
    tree: TrinomialTree = _make_tree(market, N=50)
    tree.price(option, compute_greeks=True)
    rho: float = tree.rho(option)
    assert isinstance(rho, float)
