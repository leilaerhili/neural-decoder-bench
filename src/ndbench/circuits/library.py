"""The fixed set of circuits the benchmark uses.

Noiseless memory-experiment circuits copied from quantumlib/tesseract-decoder
testdata (see scripts/vendor_circuits.py and NOTICE). Rounds = distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import stim

from ..noise import models as noise_models


@dataclass(frozen=True)
class CodeSpec:
    key: str
    family: str  # "surface", "bb" or "color"
    n: int  # data qubits
    k: int  # logical qubits
    d: int
    rounds: int


CODES = {
    "surface_d3": CodeSpec("surface_d3", "surface", 9, 1, 3, 3),
    "surface_d5": CodeSpec("surface_d5", "surface", 25, 1, 5, 5),
    "surface_d7": CodeSpec("surface_d7", "surface", 49, 1, 7, 7),
    "bb72": CodeSpec("bb72", "bb", 72, 12, 6, 6),
    "bb144": CodeSpec("bb144", "bb", 144, 12, 12, 12),
    # Triangular color code, superdense syndrome cycle (Gidney & Jones 2023).
    "color_d3": CodeSpec("color_d3", "color", 7, 1, 3, 3),
    "color_d5": CodeSpec("color_d5", "color", 19, 1, 5, 5),
    "color_d7": CodeSpec("color_d7", "color", 37, 1, 7, 7),
}


@lru_cache(maxsize=None)
def noiseless(code: str, basis: str = "Z") -> stim.Circuit:
    if code not in CODES or basis not in ("X", "Z"):
        raise KeyError(f"unknown circuit {code}_{basis}")
    text = resources.files("ndbench.circuits").joinpath("data", f"{code}_{basis}.stim").read_text()
    return stim.Circuit(text)


def noise_model(code: str, noise: str, p: float) -> noise_models.NoiseModel:
    """Noise model for `code`, using the SI1000 convention of its reference circuits."""
    # Surface and color references add DEPOLARIZE1(p) after measurement; BB references don't.
    kw = {"post_measure_depol": CODES[code].family != "bb"} if noise == "si1000" else {}
    return noise_models.get(noise, p, **kw)


def noisy(code: str, noise: str, p: float, basis: str = "Z") -> stim.Circuit:
    return noise_model(code, noise, p).apply(noiseless(code, basis))
