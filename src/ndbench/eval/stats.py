"""Statistics used throughout: intervals, LER conversions, paired tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(failures: int, shots: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and Wilson score interval for a binomial rate."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    z = stats.norm.ppf(0.5 + conf / 2)
    p = failures / shots
    denom = 1 + z**2 / shots
    centre = (p + z**2 / (2 * shots)) / denom
    half = z * np.sqrt(p * (1 - p) / shots + z**2 / (4 * shots**2)) / denom
    lo = 0.0 if failures == 0 else max(0.0, centre - half)
    hi = 1.0 if failures == shots else min(1.0, centre + half)
    return p, float(lo), float(hi)


def per_round(p_shot, rounds: int):
    """Shot-level failure probability -> per-round rate: 1 - (1 - P)^(1/R).

    Assumes independent rounds. Monotone, so it maps interval endpoints too.
    """
    return 1 - (1 - np.asarray(p_shot, dtype=float)) ** (1 / rounds)


def per_round_per_qubit(p_shot, rounds: int, k: int):
    """Per round and per logical qubit: 1 - (1 - P)^(1/(R k))."""
    return 1 - (1 - np.asarray(p_shot, dtype=float)) ** (1 / (rounds * k))


def ratio_bootstrap(fail_a: np.ndarray, fail_b: np.ndarray, n_boot: int = 2000,
                    seed: int = 0, conf: float = 0.95) -> tuple[float, float, float]:
    """LER ratio A/B on paired shots with a percentile bootstrap interval.

    Shots are resampled jointly, so the pairing is kept. Returns (ratio, lo, hi);
    inf/nan when B has no failures.
    """
    fail_a = np.asarray(fail_a, bool)
    fail_b = np.asarray(fail_b, bool)
    n = len(fail_a)
    if n != len(fail_b):
        raise ValueError("paired arrays must have equal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    fa, fb = fail_a[idx].sum(axis=1), fail_b[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = fa / fb
    point = fail_a.sum() / fail_b.sum() if fail_b.sum() else float("inf")
    lo, hi = np.nanpercentile(ratios, [50 * (1 - conf), 50 * (1 + conf)])
    return float(point), float(lo), float(hi)


def mcnemar_exact(fail_a: np.ndarray, fail_b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired per-shot failures of two decoders.

    Returns (n_a_only, n_b_only, two-sided p-value).
    """
    fail_a = np.asarray(fail_a, bool)
    fail_b = np.asarray(fail_b, bool)
    a_only = int(np.sum(fail_a & ~fail_b))
    b_only = int(np.sum(fail_b & ~fail_a))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: which hypotheses are rejected at family-wise level alpha."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    rejected, still = {}, True
    for rank, (name, p) in enumerate(items):
        still = still and p <= alpha / (m - rank)
        rejected[name] = still
    return rejected
