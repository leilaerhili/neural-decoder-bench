"""Training loop for neural decoders.

Training data is generated on the fly: every step draws a fresh batch of shots
from Stim, so the model never sees the same shot twice and there is no dataset
to store. Shots come from the "train" seed range, so they never overlap with
the validation or test shots used for scoring.

Fairness rules from docs/PLAN.md (D6) that this loop enforces:
- the budget is counted in training *samples* (shots), the same unit for every model;
- AdamW for every model (same optimizer);
- no early stopping: the final EMA weights are the model.

Built for rented (often interruptible) GPUs:
- **Resumable.** The full training state is saved to `latest.pt` every
  `save_every` steps; rerunning the same command continues from there.
- **Deterministic shot stream.** Shots come in blocks of `block_steps` batches,
  each block from its own seed. Resuming at any step reproduces exactly the
  shots an uninterrupted run would have seen.
- **Parallel sampling.** `workers` CPU processes generate blocks while the GPU
  trains (0 = sample in the main process).
- **bf16** autocast on CUDA.
- **Low-p curriculum** (from the paper): during the first `curriculum_frac` of
  training, the physical error rate ramps from p_train/4 up to p_train.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from .circuits import noisy
from .data import Sampler, seed_for
from .data.sampler import SPLIT_SIZE
from .models.cascade import CascadeConfig, CascadeSurface
from .models.layout import SurfaceLayout


@dataclass
class TrainConfig:
    code: str = "surface_d5"
    noise: str = "uniform"
    p_train: float = 0.007
    basis: str = "Z"
    model: dict = field(default_factory=lambda: CascadeConfig().to_dict())
    samples: int = 10_000_000  # total training shots (the budget)
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    curriculum_frac: float = 0.0  # 0.02 = the paper's short low-p warm-up
    seed: int = 0  # selects the training shot stream and the weight init
    device: str = "auto"
    workers: int = 0  # CPU processes generating shots in parallel
    block_steps: int = 50  # batches per seeded block of shots
    bf16: bool = True  # only used on CUDA
    log_every: int = 100
    save_every: int = 1000  # steps between resumable `latest.pt` saves
    val_shots: int = 20_000  # fixed validation set, from the "val" seed range
    checkpoints_at: list = field(default_factory=lambda: [10**6, 3 * 10**6, 10**7, 3 * 10**7, 10**8])

    @property
    def steps(self) -> int:
        return math.ceil(self.samples / self.batch)

    @property
    def tag(self) -> str:
        return f"{self.code}/{self.basis}/{self.noise}/p={self.p_train:g}"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(code: str, basis: str, model_cfg: dict) -> tuple[CascadeSurface, SurfaceLayout]:
    from .circuits import CODES, noiseless

    if CODES[code].family != "surface":
        raise NotImplementedError("Cascade for BB and color codes comes in Phase 4 / 4b")
    layout = SurfaceLayout.from_circuit(noiseless(code, basis))
    return CascadeSurface(layout, CascadeConfig(**model_cfg)), layout


# --------------------------------------------------------------------------
# Shot stream
# --------------------------------------------------------------------------

def block_seed(cfg: TrainConfig, block: int) -> int:
    """Seed for one block of training shots; always inside the train seed range."""
    base = seed_for("train", cfg.tag, cfg.seed)
    return (base + block * 1_000_003) % SPLIT_SIZE


def block_p(cfg: TrainConfig, block: int) -> float:
    """Physical error rate for a block: low-p ramp during the curriculum, then p_train."""
    ramp_steps = cfg.curriculum_frac * cfg.steps
    if ramp_steps <= 0:
        return cfg.p_train
    frac = min(1.0, block * cfg.block_steps / ramp_steps)
    return cfg.p_train * (0.25 + 0.75 * frac)


class ShotStream(IterableDataset):
    """Yields (grid uint8 (B, T, H, W), obs float32 (B, k)) batches from `start_step` on.

    Restricted to the blocks first_block + w, first_block + w + n, ... so that
    `n` of these (one per worker) together cover every block exactly once.
    """

    def __init__(self, cfg: TrainConfig, layout: SurfaceLayout, start_step: int = 0, w: int = 0, n: int = 1):
        self.cfg, self.layout, self.start_step, self.w, self.n = cfg, layout, start_step, w, n

    def __iter__(self):
        cfg = self.cfg
        first_block = self.start_step // cfg.block_steps
        last_block = (cfg.steps - 1) // cfg.block_steps
        for block in range(first_block + self.w, last_block + 1, self.n):
            circuit = noisy(cfg.code, cfg.noise, block_p(cfg, block), cfg.basis)
            sampler = Sampler(circuit, block_seed(cfg, block))
            b0 = block * cfg.block_steps
            for step in range(b0, min(b0 + cfg.block_steps, cfg.steps)):
                shots = sampler.sample(cfg.batch)
                if step < self.start_step:
                    continue  # resuming mid-block: regenerate and skip
                grid = self.layout.to_tensor(shots.dets).astype(np.uint8)
                yield torch.from_numpy(grid), torch.from_numpy(shots.obs.astype(np.float32))


def ordered_batches(cfg: TrainConfig, layout: SurfaceLayout, start_step: int = 0):
    """Training batches in exact step order, generated by `cfg.workers` processes.

    Worker w owns every n-th block; whole blocks are pulled from the workers in
    turn, so the batch order is identical for any number of workers (and on resume).
    """
    if cfg.workers <= 0:
        yield from ShotStream(cfg, layout, start_step)
        return
    n = cfg.workers
    iters = [iter(DataLoader(ShotStream(cfg, layout, start_step, w, n), batch_size=None, num_workers=1,
                             prefetch_factor=4, pin_memory=torch.cuda.is_available(),
                             multiprocessing_context="spawn"))  # fork can deadlock under torch threads
             for w in range(n)]
    first_block = start_step // cfg.block_steps
    for block in range(first_block, (cfg.steps - 1) // cfg.block_steps + 1):
        b0 = block * cfg.block_steps
        size = min(b0 + cfg.block_steps, cfg.steps) - max(b0, start_step)
        src = iters[(block - first_block) % n]
        for _ in range(size):
            yield next(src)


# --------------------------------------------------------------------------
# EMA, evaluation, checkpoints
# --------------------------------------------------------------------------

class EMA:
    """Exponential moving average of the weights (the model that gets evaluated)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Warm-up: early on, average over the few steps seen so far, so the
        # random initial weights don't dominate short runs.
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                self.shadow[k].copy_(v)


@torch.no_grad()
def logical_error_rate(model: nn.Module, grid: torch.Tensor, obs: torch.Tensor, batch: int = 4096) -> float:
    model.eval()
    wrong = 0
    for i in range(0, len(grid), batch):
        pred = model(grid[i:i + batch].float()) > 0
        wrong += int((pred != obs[i:i + batch]).any(dim=1).sum())
    return wrong / len(grid)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: Path, cfg: TrainConfig, weights: dict, samples: int) -> None:
    _atomic_save({"train_config": asdict(cfg), "weights": weights, "samples": samples}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[CascadeSurface, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    tc = ck["train_config"]
    model, _ = build_model(tc["code"], tc["basis"], tc["model"])
    model.load_state_dict(ck["weights"])
    model.to(device).eval()
    return model, ck


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(cfg: TrainConfig, out_dir: str | Path, quiet: bool = False, resume: bool = True,
          stop_after_steps: int | None = None) -> Path:
    """Train one model; writes checkpoints and a JSONL log to out_dir.

    If out_dir/latest.pt exists (and `resume`), training continues from it.
    `stop_after_steps` exits early without finishing (used to test resuming).
    Returns the final checkpoint path (or latest.pt if stopped early).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    use_bf16 = cfg.bf16 and device.type == "cuda"
    torch.manual_seed(cfg.seed)

    model, layout = build_model(cfg.code, cfg.basis, cfg.model)
    model.to(device)
    circuit = noisy(cfg.code, cfg.noise, cfg.p_train, cfg.basis)
    val = Sampler(circuit, seed_for("val", cfg.tag, 0)).sample(cfg.val_shots)
    val_grid = torch.from_numpy(layout.to_tensor(val.dets).astype(np.uint8)).to(device)
    val_obs = torch.from_numpy(val.obs).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.steps

    def lr_at(step: int) -> float:  # linear warmup, then cosine decay to 0
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * frac))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = EMA(model, cfg.ema_decay)
    start_step, elapsed_before = 0, 0.0
    latest = out / "latest.pt"
    log_path = out / "log.jsonl"
    if resume and latest.exists():
        st = torch.load(latest, map_location=device, weights_only=False)
        if st["train_config"] != asdict(cfg):
            raise ValueError(f"{latest} was made with a different config; use a new output folder")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        ema.shadow, ema.updates = st["ema"], st["ema_updates"]
        torch.set_rng_state(st["torch_rng"])
        start_step, elapsed_before = st["step"], st["elapsed"]
        if not quiet:
            print(f"resuming at step {start_step} ({start_step * cfg.batch:,} samples)", flush=True)
    else:
        log_path.write_text("")
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    def save_state(step: int) -> None:
        _atomic_save({"train_config": asdict(cfg), "model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "ema": ema.shadow, "ema_updates": ema.updates,
                      "torch_rng": torch.get_rng_state(), "step": step,
                      "elapsed": elapsed_before + time.time() - t0}, latest)

    loss_fn = nn.BCEWithLogitsLoss()
    pending = sorted(c for c in cfg.checkpoints_at if start_step * cfg.batch < c < cfg.samples)
    t0 = time.time()
    running, n_running = 0.0, 0
    batches = ordered_batches(cfg, layout, start_step)

    for step in range(start_step, steps):
        model.train()
        grid, target = next(batches)
        grid = grid.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(grid)
        loss = loss_fn(logits.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        running += loss.item()
        n_running += 1

        done = step + 1
        seen = done * cfg.batch
        last = done == steps
        if done % cfg.log_every == 0 or last:
            secs = elapsed_before + time.time() - t0
            row = {"step": done, "samples": seen, "loss": running / n_running,
                   "lr": sched.get_last_lr()[0], "p": block_p(cfg, step // cfg.block_steps),
                   "seconds": round(secs, 1),
                   "samples_per_s": round((done - start_step) * cfg.batch / max(time.time() - t0, 1e-9))}
            running, n_running = 0.0, 0
            if done % (cfg.log_every * 5) == 0 or last:
                row["val_ler_ema"] = _ema_val(model, ema, val_grid, val_obs)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if not quiet:
                print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
        while pending and seen >= pending[0]:
            save_checkpoint(out / f"ckpt_{pending.pop(0):.0e}.pt", cfg, ema.shadow, seen)
        if done % cfg.save_every == 0 and not last:
            save_state(done)
        if stop_after_steps is not None and done - start_step >= stop_after_steps and not last:
            save_state(done)
            return latest

    final = out / "final.pt"
    save_checkpoint(final, cfg, ema.shadow, steps * cfg.batch)
    if latest.exists():
        latest.unlink()
    return final


def _ema_val(model: nn.Module, ema: EMA, grid: torch.Tensor, obs: torch.Tensor) -> float:
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.shadow)
    ler = logical_error_rate(model, grid, obs)
    model.load_state_dict(backup)
    return ler
