"""Run reference decoders over a set of cells and append results to a JSONL file.

Each cell uses its fixed test seed, so these shots are the same first shots any
neural decoder sees later on that cell (paired comparisons).

    python scripts/reference_sweep.py --codes surface_d5 surface_d7 \
        --decoders pymatching --max-shots 1000000 --max-failures 1000 \
        --out results/reference_surface.jsonl

Evaluation p values per code come from configs/grid.toml.
Cells already present in the output file (same decoder, settings, cell) are skipped,
so an interrupted sweep can simply be rerun.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from ndbench.decoders import REGISTRY
from ndbench.eval.runner import append_jsonl, evaluate

ROOT = Path(__file__).resolve().parents[1]


def done_cells(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.add((json.dumps(r["decoder"], sort_keys=True), r["code"], r["basis"], r["noise"], r["p"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", required=True)
    ap.add_argument("--noises", nargs="+", default=["uniform", "si1000"])
    ap.add_argument("--decoders", nargs="+", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--preset", default=None, help="tesseract preset")
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--max-shots", type=int, default=100_000)
    ap.add_argument("--max-failures", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1_000)
    ap.add_argument("--p", type=float, nargs="*", default=None, help="override the grid's p values")
    ap.add_argument("--out", default="results/reference.jsonl")
    a = ap.parse_args()

    grid = tomllib.loads((ROOT / "configs" / "grid.toml").read_text())
    out = Path(a.out)
    seen = done_cells(out)
    for name in a.decoders:
        kwargs = {"preset": a.preset} if (a.preset and name == "tesseract") else {}
        for code in a.codes:
            for noise in a.noises:
                for p in (a.p or grid["noise"]["eval_p"][code]):
                    dec = REGISTRY[name](**kwargs)
                    key = (json.dumps(dec.describe(), sort_keys=True), code, a.basis, noise, p)
                    if key in seen:
                        continue
                    r = evaluate(dec, code, noise, p, a.basis, max_shots=a.max_shots,
                                 max_failures=a.max_failures, batch=a.batch)
                    append_jsonl(r, out)
                    print(f"{name:10s} {code:11s} {noise:7s} p={p:<6g} {r.failures:5d}/{r.shots:<8d} "
                          f"LER/shot={r.ler_shot:.3e} [{r.ler_shot_lo:.2e}, {r.ler_shot_hi:.2e}] "
                          f"{r.seconds_per_shot * 1e3:.3f} ms/shot", flush=True)


if __name__ == "__main__":
    main()
