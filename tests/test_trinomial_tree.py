import datetime as dt
import pytest
import random
from pricing.trinomial_tree import TrinomialTree
from pricing import Market, Option

@pytest.mark.parametrize(
    "S0, r, sigma, dividend, N, epsilon",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(0, 10), random.randint(10, 100), 1e-6) for _ in range(5)]
)
def test_trinomial_tree_initialization(S0, r, sigma, dividend, N, epsilon):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=dividend)
    tree = TrinomialTree(market=market, N=N, pruning=True, epsilon=epsilon, pricing_date=dt.datetime(2025, 1, 1))
    assert tree.market == market
    assert tree.N == N
    assert tree.pruning is True
    assert tree.epsilon == epsilon
    assert tree.pricing_date == dt.datetime(2025, 1, 1)

@pytest.mark.parametrize(
    "S0, r, sigma, dividend, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(0, 10), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_compute_parameters(S0, r, sigma, dividend, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=dividend, dividend_date=dt.datetime(2025, 6, 1))
    tree = TrinomialTree(market=market, N=5, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree.option = option
    tree._compute_parameters(S=market.S0, validate=True)
    assert 0 <= tree.p_down <= 1
    assert 0 <= tree.p_up <= 1
    assert 0 <= tree.p_mid <= 1
    assert tree.p_down + tree.p_up + tree.p_mid == pytest.approx(1.0)
    assert tree.alpha > 1.0
    assert tree.delta_t > 0

@pytest.mark.parametrize(
    "S0, r, sigma, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_price(S0, r, sigma, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=0.0)
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree = TrinomialTree(market=market, N=50, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    price = tree.price(option, method="backward", build_tree=True)
    assert price > 0

@pytest.mark.parametrize(
    "S0, r, sigma, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_delta(S0, r, sigma, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=0.0)
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree = TrinomialTree(market=market, N=50, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    tree.price(option, compute_greeks=True)
    delta = tree.delta()
    assert isinstance(delta, float)

@pytest.mark.parametrize(
    "S0, r, sigma, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_gamma(S0, r, sigma, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=0.0)
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree = TrinomialTree(market=market, N=50, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    tree.price(option, compute_greeks=True)
    gamma = tree.gamma()
    assert isinstance(gamma, float)

@pytest.mark.parametrize(
    "S0, r, sigma, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_vega(S0, r, sigma, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=0.0)
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree = TrinomialTree(market=market, N=50, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    tree.price(option, compute_greeks=True)
    vega = tree.vega(option)
    assert isinstance(vega, float)

@pytest.mark.parametrize(
    "S0, r, sigma, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_vanna(S0, r, sigma, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=0.0)
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree = TrinomialTree(market=market, N=50, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    tree.price(option, compute_greeks=True)
    vanna = tree.vanna(option)
    assert isinstance(vanna, float)

@pytest.mark.parametrize(
    "S0, r, sigma, K, maturity",
    [(random.uniform(80, 120), random.uniform(0.01, 0.05), random.uniform(0.1, 0.3), random.uniform(90, 110), dt.datetime(2025, 7, 1)) for _ in range(5)]
)
def test_trinomial_tree_rho(S0, r, sigma, K, maturity):
    market = Market(S0=S0, r=r, sigma=sigma, dividend=0.0)
    option = Option(K=K, option_type="call", maturity=maturity, option_class="european")
    tree = TrinomialTree(market=market, N=50, pruning=False, epsilon=1e-7, pricing_date=dt.datetime(2025, 1, 1))
    tree.price(option, compute_greeks=True)
    rho = tree.rho(option)
    assert isinstance(rho, float)