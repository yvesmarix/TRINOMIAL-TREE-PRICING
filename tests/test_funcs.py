
import math
import numpy as np
import pytest

from pricing import funcs

def test_compute_forward_matches_formula():
    S, r, dt_ = 123.45, 0.037, 0.4
    assert funcs.compute_forward(S, r, dt_) == pytest.approx(S * math.exp(r * dt_))

@pytest.mark.parametrize("S,r,sigma,dt_,alpha", [
    (100.0, 0.01, 0.20, 0.25, 1/3),
    (120.0, 0.03, 0.15, 0.10, 0.25),
    (80.0,  0.02, 0.30, 0.50, 0.5),
])
def test_compute_probabilities_sum_to_one_and_in_01_with_coherent_inputs(S, r, sigma, dt_, alpha):
    forward = funcs.compute_forward(S, r, dt_)
    variance = funcs.compute_variance(S, r, dt_, sigma)
    p_down, p_up, p_mid = funcs.compute_probabilities(forward, forward, variance, alpha)
    for p in (p_down, p_up, p_mid):
        assert 0.0 <= p <= 1.0
    assert p_down + p_up + p_mid == pytest.approx(1.0)

def test_variance_and_probabilities_pipeline_is_consistent():
    # Typical market numbers
    S0, r, sigma, dt_ = 100.0, 0.02, 0.25, 0.5
    variance = funcs.compute_variance(S0, r, dt_, sigma)
    forward = funcs.compute_forward(S0, r, dt_)
    # no dividend case; choose alpha=1/3 (trinomial "equal step" style)
    p_down, p_up, p_mid = funcs.compute_probabilities(forward, forward, variance, 1/3)
    # they should form a valid distribution
    class _T: pass
    t = _T()
    t.p_down, t.p_up, t.p_mid = p_down, p_up, p_mid
    assert funcs.probas_valid(t)

def test_iter_column_yields_up_then_down():
    # Build a small column of nodes with up chain length 2 and down chain length 1
    import types
    Node = types.SimpleNamespace  # simple stand-in
    mid = Node(S=100, proba=1.0, up=None, down=None)
    up1 = Node(S=110, proba=0.0, up=None, down=None)
    up2 = Node(S=120, proba=0.0, up=None, down=None)
    down1 = Node(S=90,  proba=0.0, up=None, down=None)
    mid.up = up1
    up1.up = up2
    mid.down = down1

    seq = list(funcs.iter_column(mid))
    # order: mid, up1, up2, then down1
    assert [n.S for n in seq] == [100, 110, 120, 90]
