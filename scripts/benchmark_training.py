"""Speed test for a (rented) machine: how fast does training run, and what will it cost?

Run this first on any new GPU machine. It takes a few minutes and prints, for
each setting, training throughput (samples/s) and the projected time and cost
of a full 1e8-sample run.

    python scripts/benchmark_training.py --price-per-hour 0.70

What it measures:
- **Sampling only:** how many shots/s one CPU worker can generate (Stim +
  turning shots into pictures). If training is faster than workers x this
  number, the GPU waits for data: add workers.
- **Training:** full training steps (sampling + GPU) with the given workers.
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch

from ndbench.circuits import noiseless
from ndbench.models.layout import SurfaceLayout
from ndbench.training import TrainConfig, build_model, ordered_batches, pick_device


def sampling_rate(code: str, batch: int, seconds: float = 5.0) -> float:
    cfg = TrainConfig(code=code, batch=batch, samples=batch * 10**6, workers=0, model={"hidden": 8, "layers": 1})
    lay = SurfaceLayout.from_circuit(noiseless(code, "Z"))
    it = ordered_batches(cfg, lay)
    next(it)
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        next(it)
        n += batch
    return n / (time.time() - t0)


def training_rate(code: str, hidden: int, layers: int, batch: int, workers: int, steps: int, bf16: bool) -> float:
    device = pick_device("auto")
    cfg = TrainConfig(code=code, batch=batch, samples=batch * (steps + 20), workers=workers, bf16=bf16,
                      model={"hidden": hidden, "layers": layers})
    model, lay = build_model(code, "Z", cfg.model)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    use_bf16 = bf16 and device.type == "cuda"
    it = ordered_batches(cfg, lay)

    def step():
        g, y = next(it)
        g, y = g.to(device, non_blocking=True).float(), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(g)
        loss = loss_fn(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(10):  # warm-up (worker start-up, CUDA kernels)
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return steps * batch / (time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", default=["surface_d5", "surface_d7"])
    ap.add_argument("--hidden", nargs="+", type=int, default=[128])
    ap.add_argument("--layers", type=int, default=None, help="default: d (the code distance)")
    ap.add_argument("--batches", nargs="+", type=int, default=[1024, 4096])
    ap.add_argument("--workers", nargs="+", type=int, default=None, help="default: 0 and (CPUs - 2)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--price-per-hour", type=float, default=None, help="machine price, for cost estimates")
    ap.add_argument("--budget-samples", type=float, default=1e8)
    a = ap.parse_args()

    import os
    cpus = os.cpu_count() or 2
    workers = a.workers or sorted({0, max(1, cpus - 2)})
    dev = pick_device("auto")
    name = torch.cuda.get_device_name() if dev.type == "cuda" else dev.type
    print(f"device: {name}   CPUs: {cpus}   torch {torch.__version__}\n")

    for code in a.codes:
        print(f"{code}: one worker generates {sampling_rate(code, 1024):,.0f} shots/s")
    print()
    print(f"{'code':11s} {'H':>4s} {'L':>3s} {'batch':>6s} {'workers':>7s} {'samples/s':>10s} "
          f"{'hours/1e8':>9s}" + ("  cost/1e8" if a.price_per_hour else ""))
    for code, h, b, w in itertools.product(a.codes, a.hidden, a.batches, workers):
        layers = a.layers or int(code.split("_d")[-1])
        try:
            rate = training_rate(code, h, layers, b, w, a.steps, not a.no_bf16)
        except RuntimeError as e:  # e.g. out of memory
            print(f"{code:11s} {h:4d} {layers:3d} {b:6d} {w:7d}  failed: {str(e)[:60]}")
            continue
        hours = a.budget_samples / rate / 3600
        cost = f"  ${hours * a.price_per_hour:8.2f}" if a.price_per_hour else ""
        print(f"{code:11s} {h:4d} {layers:3d} {b:6d} {w:7d} {rate:10,.0f} {hours:9.2f}{cost}", flush=True)


if __name__ == "__main__":
    main()
