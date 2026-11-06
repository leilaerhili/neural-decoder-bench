import numpy as np
import pytest

from ndbench.eval import holm, mcnemar_exact, per_round, per_round_per_qubit, ratio_bootstrap, wilson


def test_wilson_known_value():
    p, lo, hi = wilson(10, 100)
    assert p == 0.1
    assert lo == pytest.approx(0.0552, abs=1e-4)
    assert hi == pytest.approx(0.1744, abs=1e-4)


def test_wilson_edges():
    assert wilson(0, 1000)[1] == 0.0
    assert wilson(1000, 1000)[2] == 1.0


def test_per_round_inverts():
    pr = 1e-4
    p_shot = 1 - (1 - pr) ** 6
    assert per_round(p_shot, 6) == pytest.approx(pr)
    assert per_round_per_qubit(p_shot, 6, 12) < pr


def test_ratio_bootstrap():
    rng = np.random.default_rng(0)
    b = rng.random(20000) < 0.02
    a = b | (rng.random(20000) < 0.02)  # A fails whenever B does, plus more
    r, lo, hi = ratio_bootstrap(a, b, n_boot=500)
    assert lo < r < hi and 1.6 < r < 2.4


def test_mcnemar_and_holm():
    a = np.array([1] * 8 + [0] * 2, bool)
    assert mcnemar_exact(a, np.zeros(10, bool))[2] < 0.01
    assert holm({"h1": 0.001, "h2": 0.03, "h3": 0.04}) == {"h1": True, "h2": False, "h3": False}
