"""Evaluate a reference decoder on one cell and append the result to a JSONL file.

Example:
    python scripts/eval_reference.py --decoder relay_bp --code bb72 --noise si1000 \
        --p 0.003 --max-shots 10000 --out results/reference.jsonl
"""

from __future__ import annotations

import argparse

from ndbench.decoders import REGISTRY
from ndbench.eval.runner import append_jsonl, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=sorted(REGISTRY), required=True)
    ap.add_argument("--preset", default=None, help="tesseract preset, e.g. tesseract-short-beam")
    ap.add_argument("--code", required=True)
    ap.add_argument("--noise", choices=["uniform", "si1000"], required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--max-shots", type=int, default=10_000)
    ap.add_argument("--max-failures", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1_000)
    ap.add_argument("--out", default="results/reference.jsonl")
    a = ap.parse_args()

    kwargs = {"preset": a.preset} if a.preset else {}
    dec = REGISTRY[a.decoder](**kwargs)
    r = evaluate(dec, a.code, a.noise, a.p, a.basis, max_shots=a.max_shots,
                 max_failures=a.max_failures, batch=a.batch)
    append_jsonl(r, a.out)
    print(f"{a.decoder} {a.code} {a.noise} p={a.p}: {r.failures}/{r.shots} "
          f"LER/shot={r.ler_shot:.3e} [{r.ler_shot_lo:.2e}, {r.ler_shot_hi:.2e}] "
          f"per-round-per-qubit={r.ler_round_per_qubit:.3e} {r.seconds_per_shot * 1e3:.2f} ms/shot")


if __name__ == "__main__":
    main()
