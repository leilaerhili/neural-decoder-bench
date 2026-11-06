import numpy as np
import pytest

from ndbench.circuits import noisy
from ndbench.data import Sampler, seed_for, split_of
from ndbench.decoders import PyMatchingDecoder
from ndbench.eval.runner import append_jsonl, evaluate


def test_seed_splits_disjoint():
    tags = [f"bb72/Z/si1000/p={p}" for p in (0.001, 0.002, 0.003)]
    train = {seed_for("train", t, i) for t in tags for i in range(1000)}
    test = {seed_for("test", t, i) for t in tags for i in range(1000)}
    assert not train & test
    assert all(split_of(s) == "test" for s in test)
    with pytest.raises(ValueError):
        seed_for("holdout", "x")


def test_sampler_deterministic():
    c = noisy("surface_d3", "si1000", 0.003)
    a = Sampler(c, 123).sample(500)
    b = Sampler(c, 123).sample(500)
    np.testing.assert_array_equal(a.dets, b.dets)
    np.testing.assert_array_equal(a.obs, b.obs)
    assert a.dets.dtype == bool and a.dets.shape == (500, c.num_detectors)


def test_runner_paired_shots_and_sane_ler(tmp_path):
    """PyMatching beats 'always predict no flip' on the surface code, and two
    runs with the same seed see the same shots."""

    class Trivial(PyMatchingDecoder):
        name = "trivial"

        def decode(self, dets):
            return np.zeros((len(dets), 1), bool)

    kw = dict(max_shots=20_000, max_failures=10**9, batch=5_000)
    r_pm, f_pm = evaluate(PyMatchingDecoder(), "surface_d3", "si1000", 0.003, keep_failures=True, **kw)
    r_tr, f_tr = evaluate(Trivial(), "surface_d3", "si1000", 0.003, keep_failures=True, **kw)
    assert r_pm.seed == r_tr.seed and len(f_pm) == len(f_tr) == 20_000
    assert r_pm.ler_shot < 0.5 * r_tr.ler_shot
    assert 0 < r_pm.ler_round_per_qubit <= r_pm.ler_shot
    append_jsonl(r_pm, tmp_path / "r.jsonl")
    assert (tmp_path / "r.jsonl").read_text().count("\n") == 1


@pytest.mark.slow
def test_reference_decoders_run_on_bb72():
    from ndbench.decoders import RelayBPDecoder, TesseractDecoder

    for dec in (RelayBPDecoder(), TesseractDecoder("tesseract-short-beam")):
        r = evaluate(dec, "bb72", "si1000", 0.002, max_shots=50, max_failures=10**9, batch=50)
        assert r.shots == 50
