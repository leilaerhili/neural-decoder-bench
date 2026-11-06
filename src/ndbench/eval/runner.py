"""Evaluate one decoder on one cell (code, noise, p, basis) with paired shots.

Shots are drawn in batches from a fixed test seed until `max_failures` logical
failures or `max_shots` shots, whichever comes first. Because the seed fixes
the shot stream, two decoders run with the same `max_shots` see identical
shots (use a fixed shot count, not a failure target, for paired tests).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..circuits.library import CODES, noisy
from ..data.sampler import Sampler, seed_for
from ..decoders.base import Decoder
from .stats import per_round, per_round_per_qubit, wilson


@dataclass
class CellResult:
    code: str
    noise: str
    p: float
    basis: str
    decoder: dict
    seed: int
    shots: int
    failures: int
    ler_shot: float
    ler_shot_lo: float
    ler_shot_hi: float
    ler_round: float
    ler_round_per_qubit: float
    decode_seconds: float
    seconds_per_shot: float
    host: dict = field(default_factory=dict)


def provenance() -> dict:
    """Git commit and package versions, recorded with every result (PLAN.md D10)."""
    import subprocess
    from importlib import metadata

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, cwd=Path(__file__).parent, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    versions = {}
    for pkg in ("stim", "numpy", "pymatching", "tesseract-decoder", "relay-bp", "torch"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return {"machine": platform.machine(), "python": platform.python_version(),
            "git_commit": commit + ("-dirty" if dirty else ""), "versions": versions}


def cell_tag(code: str, noise: str, p: float, basis: str) -> str:
    return f"{code}/{basis}/{noise}/p={p:g}"


def evaluate(decoder: Decoder, code: str, noise: str, p: float, basis: str = "Z", *,
             max_shots: int = 100_000, max_failures: int = 100, batch: int = 10_000,
             seed: int | None = None, keep_failures: bool = False):
    """Run the decoder on one cell. Returns a CellResult (and per-shot failures if asked)."""
    spec = CODES[code]
    circuit = noisy(code, noise, p, basis)
    seed = seed_for("test", cell_tag(code, noise, p, basis)) if seed is None else seed
    decoder.prepare(circuit)
    sampler = Sampler(circuit, seed)

    shots = failures = 0
    decode_time = 0.0
    fail_log = []
    while shots < max_shots and failures < max_failures:
        n = min(batch, max_shots - shots)
        data = sampler.sample(n)
        t0 = time.perf_counter()
        pred = decoder.decode(data.dets)
        decode_time += time.perf_counter() - t0
        fail = np.any(pred != data.obs, axis=1)
        failures += int(fail.sum())
        shots += n
        if keep_failures:
            fail_log.append(fail)

    ler, lo, hi = wilson(failures, shots)
    res = CellResult(
        code=code, noise=noise, p=p, basis=basis, decoder=decoder.describe(), seed=seed,
        shots=shots, failures=failures, ler_shot=ler, ler_shot_lo=lo, ler_shot_hi=hi,
        ler_round=float(per_round(ler, spec.rounds)),
        ler_round_per_qubit=float(per_round_per_qubit(ler, spec.rounds, spec.k)),
        decode_seconds=decode_time, seconds_per_shot=decode_time / shots,
        host=provenance(),
    )
    if keep_failures:
        return res, np.concatenate(fail_log)
    return res


def append_jsonl(result: CellResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
