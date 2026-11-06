"""Seeded syndrome sampling.

Every (circuit, seed) pair yields the same shots, so all decoders in a
comparison decode identical data. Seeds for training, validation and test come
from disjoint ranges.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import stim

SPLIT_SIZE = 10**9
SPLITS = {"train": 0, "val": 1, "test": 2}


def seed_for(split: str, tag: str, index: int = 0) -> int:
    """Deterministic seed in the range reserved for `split`.

    `tag` names the experiment cell (e.g. "bb72/si1000/p=0.003"); different
    tags get different streams within the split.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if not 0 <= index < 1000:
        raise ValueError("index must be in [0, 1000)")
    return SPLITS[split] * SPLIT_SIZE + (zlib.crc32(tag.encode()) % 10**6) * 1000 + index


def split_of(seed: int) -> str:
    return {v: k for k, v in SPLITS.items()}[seed // SPLIT_SIZE]


@dataclass
class Shots:
    dets: np.ndarray  # (shots, num_detectors) bool
    obs: np.ndarray  # (shots, num_observables) bool


class Sampler:
    """Wraps a compiled Stim detector sampler with a fixed seed."""

    def __init__(self, circuit: stim.Circuit, seed: int):
        self.seed = seed
        self._sampler = circuit.compile_detector_sampler(seed=seed)

    def sample(self, shots: int) -> Shots:
        dets, obs = self._sampler.sample(shots, separate_observables=True)
        return Shots(dets.astype(bool), obs.astype(bool))
