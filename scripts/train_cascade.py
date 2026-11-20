"""Train Cascade on one (code, noise, p_train) setting.

Quick check on a laptop (a few minutes on CPU):
    python scripts/train_cascade.py --code surface_d3 --p-train 0.005 \
        --samples 2e6 --hidden 64 --layers 3 --out runs/cascade_d3_smoke

Real run on a GPU (Phase 2); see docs/GPU_RUNBOOK.md:
    python scripts/train_cascade.py --code surface_d5 --p-train 0.007 \
        --samples 1e8 --batch 4096 --workers 6 --curriculum 0.02 \
        --out runs/cascade_d5_uniform_s0

If the machine stops, rerun the exact same command: it resumes from
<out>/latest.pt.
"""

from __future__ import annotations

import argparse

from ndbench.training import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="surface_d5")
    ap.add_argument("--noise", default="uniform", choices=["uniform", "si1000"])
    ap.add_argument("--p-train", type=float, default=0.007)
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--samples", type=float, default=1e7)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0, help="CPU processes generating shots")
    ap.add_argument("--curriculum", type=float, default=0.0, help="fraction of training on a low-p ramp (paper: 0.02)")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--save-every", type=int, default=1000, help="steps between resumable saves")
    ap.add_argument("--no-resume", action="store_true", help="refuse to continue from latest.pt")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = TrainConfig(code=a.code, noise=a.noise, p_train=a.p_train, basis=a.basis,
                      model={"hidden": a.hidden, "layers": a.layers}, samples=int(a.samples),
                      batch=a.batch, lr=a.lr, seed=a.seed, device=a.device, log_every=a.log_every,
                      workers=a.workers, curriculum_frac=a.curriculum, bf16=not a.no_bf16,
                      save_every=a.save_every)
    final = train(cfg, a.out, resume=not a.no_resume)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
