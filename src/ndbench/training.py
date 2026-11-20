"""Training loop for neural decoders.

Training data is generated on the fly: every step draws a fresh batch of shots
from Stim, so the model never sees the same shot twice and there is no dataset
to store. Shots come from the "train" seed range, so they never overlap with
the validation or test shots used for scoring.

Fairness rules from docs/PLAN.md (D6) that this loop enforces:
- the budget is counted in training *samples* (shots), the same unit for every model;
- AdamW for every model (same optimizer);
- no early stopping: the final EMA weights are the model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .circuits import noisy
from .data import Sampler, seed_for
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
    seed: int = 0  # selects the training shot stream and the weight init
    device: str = "auto"
    log_every: int = 100
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
        pred = model(grid[i:i + batch]) > 0
        wrong += int((pred != obs[i:i + batch]).any(dim=1).sum())
    return wrong / len(grid)


def train(cfg: TrainConfig, out_dir: str | Path, quiet: bool = False) -> Path:
    """Train one model; writes checkpoints and a JSONL log to out_dir. Returns the final checkpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    model, layout = build_model(cfg.code, cfg.basis, cfg.model)
    model.to(device)
    circuit = noisy(cfg.code, cfg.noise, cfg.p_train, cfg.basis)
    train_sampler = Sampler(circuit, seed_for("train", cfg.tag, cfg.seed))
    val = Sampler(circuit, seed_for("val", cfg.tag, 0)).sample(cfg.val_shots)
    val_grid = torch.from_numpy(layout.to_tensor(val.dets)).to(device)
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
    loss_fn = nn.BCEWithLogitsLoss()
    log_path = out / "log.jsonl"
    log_path.write_text("")
    pending = sorted(c for c in cfg.checkpoints_at if c < cfg.samples)
    t0 = time.time()
    running = 0.0

    for step in range(steps):
        model.train()
        shots = train_sampler.sample(cfg.batch)
        grid = torch.from_numpy(layout.to_tensor(shots.dets)).to(device)
        target = torch.from_numpy(shots.obs.astype(np.float32)).to(device)
        loss = loss_fn(model(grid), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        running += loss.item()

        seen = (step + 1) * cfg.batch
        last = step == steps - 1
        if (step + 1) % cfg.log_every == 0 or last:
            row = {"step": step + 1, "samples": seen, "loss": running / ((step % cfg.log_every) + 1),
                   "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1),
                   "samples_per_s": round(seen / (time.time() - t0))}
            running = 0.0
            if (step + 1) % (cfg.log_every * 5) == 0 or last:
                row["val_ler_ema"] = _ema_val(model, ema, val_grid, val_obs)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if not quiet:
                print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
        while pending and seen >= pending[0]:
            save_checkpoint(out / f"ckpt_{pending.pop(0):.0e}.pt", cfg, ema.shadow, seen)
    final = out / "final.pt"
    save_checkpoint(final, cfg, ema.shadow, steps * cfg.batch)
    return final


def _ema_val(model: nn.Module, ema: EMA, grid: torch.Tensor, obs: torch.Tensor) -> float:
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.shadow)
    ler = logical_error_rate(model, grid, obs)
    model.load_state_dict(backup)
    return ler


def save_checkpoint(path: Path, cfg: TrainConfig, weights: dict, samples: int) -> None:
    torch.save({"train_config": asdict(cfg), "weights": weights, "samples": samples}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[CascadeSurface, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    tc = ck["train_config"]
    model, _ = build_model(tc["code"], tc["basis"], tc["model"])
    model.load_state_dict(ck["weights"])
    model.to(device).eval()
    return model, ck
