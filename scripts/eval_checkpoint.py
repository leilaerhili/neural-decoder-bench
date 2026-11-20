"""Score a trained neural decoder on one cell, next to PyMatching, on the same test shots.

    python scripts/eval_checkpoint.py runs/cascade_d3_smoke/final.pt \
        --noise uniform --p 0.003 --shots 100000
"""

from __future__ import annotations

import argparse

from ndbench.decoders import PyMatchingDecoder
from ndbench.decoders.neural import NeuralDecoder
from ndbench.eval import ratio_bootstrap
from ndbench.eval.runner import append_jsonl, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--noise", default=None, help="default: the noise the model was trained on")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--out", default="results/neural.jsonl")
    a = ap.parse_args()

    nn_dec = NeuralDecoder(a.checkpoint)
    tc = nn_dec.train_config
    noise = a.noise or tc["noise"]
    kw = dict(max_shots=a.shots, max_failures=10**12, batch=10_000, keep_failures=True)
    r_nn, f_nn = evaluate(nn_dec, tc["code"], noise, a.p, tc["basis"], **kw)
    r_pm, f_pm = evaluate(PyMatchingDecoder(), tc["code"], noise, a.p, tc["basis"], **kw)
    append_jsonl(r_nn, a.out)
    append_jsonl(r_pm, a.out)
    ratio, lo, hi = ratio_bootstrap(f_nn, f_pm, n_boot=1000)
    for r in (r_nn, r_pm):
        print(f"{r.decoder['name']:10s} {r.code} {noise} p={a.p}: {r.failures}/{r.shots} "
              f"LER/shot={r.ler_shot:.3e} [{r.ler_shot_lo:.2e}, {r.ler_shot_hi:.2e}]")
    print(f"Cascade / PyMatching error ratio: {ratio:.2f} [{lo:.2f}, {hi:.2f}] (below 1 = Cascade better)")


if __name__ == "__main__":
    main()
