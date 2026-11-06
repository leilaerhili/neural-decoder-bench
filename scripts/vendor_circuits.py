"""Copy the circuits this benchmark uses out of a tesseract-decoder checkout.

Circuits come from quantumlib/tesseract-decoder (Apache-2.0), testdata/.
We store them *noiseless* (stim's `without_noise()`); noise models are applied
by `ndbench.noise`. A few original SI1000 files are kept under tests/data to
check that our SI1000 implementation reproduces them exactly.

Usage:
    git clone --depth 1 https://github.com/quantumlib/tesseract-decoder /path/to/td
    python scripts/vendor_circuits.py /path/to/td
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import stim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "ndbench" / "circuits" / "data"
VALIDATION = ROOT / "tests" / "data"

# (our name, testdata subdir, filename regex). {B} is the basis, {P} the p value.
WANTED = {
    **{
        f"surface_d{d}_{b}": ("surfacecodes", rf"^r={d},d={d},p={{P}},noise=si1000,c=surface_code_{b},")
        for d in (3, 5, 7)
        for b in ("X", "Z")
    },
    **{
        f"bb72_{b}": ("bivariatebicyclecodes",
                      rf"^r=6,d=6,p={{P}},noise=si1000,c=bivariate_bicycle_{b},nkd=\[\[72,12,6\]\]")
        for b in ("X", "Z")
    },
    **{
        f"bb144_{b}": ("bivariatebicyclecodes",
                       rf"^r=12,d=12,p={{P}},noise=si1000,c=bivariate_bicycle_{b},nkd=\[\[144,12,12\]\]")
        for b in ("X", "Z")
    },
    # Superdense color-code cycle (Gidney & Jones 2023, arXiv:2312.08813).
    **{
        f"color_d{d}_{b}": ("colorcodes", rf"^r={d},d={d},p={{P}},noise=si1000,c=superdense_color_code_{b},")
        for d in (3, 5, 7)
        for b in ("X", "Z")
    },
}
VALIDATION_P = ("0.0005", "0.002")
VALIDATION_NAMES = ("surface_d3_Z", "surface_d5_X", "bb72_Z", "bb144_Z", "color_d5_X", "color_d7_Z")


def find(td: Path, subdir: str, pattern: str, p: str) -> Path:
    rx = re.compile(pattern.replace("{P}", re.escape(p)))
    hits = [f for f in (td / "testdata" / subdir).iterdir() if rx.search(f.name)]
    if len(hits) != 1:
        raise SystemExit(f"expected 1 match for {pattern} p={p}, got {[h.name for h in hits]}")
    return hits[0]


def main(td: str) -> None:
    td = Path(td)
    OUT.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    for name, (subdir, pattern) in WANTED.items():
        src = find(td, subdir, pattern, "0.001")
        noiseless = stim.Circuit.from_file(src).without_noise()
        header = f"# {name}: noiseless copy of tesseract-decoder testdata/{subdir}/{src.name}\n"
        (OUT / f"{name}.stim").write_text(header + str(noiseless) + "\n")
        if name in VALIDATION_NAMES:
            for p in VALIDATION_P:
                v = find(td, subdir, pattern, p)
                (VALIDATION / f"{name}_si1000_p{p}.stim").write_text(v.read_text())
        print(f"{name}: {noiseless.num_qubits} qubits, {noiseless.num_detectors} detectors, "
              f"{noiseless.num_observables} observables")


if __name__ == "__main__":
    main(sys.argv[1])
