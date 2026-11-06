"""Our noise models applied to noiseless circuits must reproduce the reference
SI1000 circuits from tesseract-decoder exactly (same detector error model)."""

from pathlib import Path

import numpy as np
import pytest
import stim

from ndbench.circuits import CODES, noiseless, noisy
from ndbench.noise import si1000, uniform

DATA = Path(__file__).parent / "data"
REFERENCES = sorted(DATA.glob("*_si1000_p*.stim"))


def _parse(path: Path):
    name, p = path.stem.split("_si1000_p")
    code, basis = name.rsplit("_", 1)
    return code, basis, float(p)


def test_references_present():
    assert len(REFERENCES) == 12  # surface, BB and color code references


@pytest.mark.parametrize("path", REFERENCES, ids=lambda p: p.stem)
def test_si1000_reproduces_reference(path):
    code, basis, p = _parse(path)
    ref = stim.Circuit.from_file(path).flattened()
    ours = noisy(code, "si1000", p, basis)
    assert ours.detector_error_model().approx_equals(ref.detector_error_model(), atol=1e-12)


@pytest.mark.parametrize("code", list(CODES))
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_circuits_load_and_have_expected_observables(code, basis):
    c = noiseless(code, basis)
    assert c.num_observables == CODES[code].k
    # Noiseless circuit: every detector is deterministic and no shot fails.
    dets, obs = c.compile_detector_sampler(seed=0).sample(16, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_noise_changes_only_noise():
    base = noiseless("surface_d3", "Z")
    for model in (uniform(0.003), si1000(0.003)):
        assert model.apply(base).without_noise() == base.flattened().without_noise()


def test_uniform_rates():
    c = uniform(0.004).apply(noiseless("surface_d3", "Z"))
    probs = {round(a, 12) for inst in c for a in inst.gate_args_copy()
             if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "M")}
    assert probs == {0.004}


def test_si1000_rates():
    p = 0.002
    c = si1000(p).apply(noiseless("surface_d3", "Z"))
    probs = {round(a, 12) for inst in c for a in inst.gate_args_copy()
             if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "M")}
    assert probs == {round(x, 12) for x in (p / 10, p, 2 * p, 5 * p)}


@pytest.mark.parametrize("noise", ["uniform", "si1000"])
def test_error_rate_increases_with_p(noise):
    """Sanity: more physical noise -> more detection events."""
    rates = []
    for p in (0.001, 0.004):
        d = noisy("bb72", noise, p).compile_detector_sampler(seed=1).sample(2000)
        rates.append(d.mean())
    assert rates[0] < rates[1]
