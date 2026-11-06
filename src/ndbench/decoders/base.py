"""Common decoder interface.

Every decoder, classical or neural, is used the same way:

    dec = SomeDecoder(**options)
    dec.prepare(circuit)                    # build from circuit / DEM, or load weights
    pred = dec.decode(dets)                 # (shots, D) bool -> (shots, O) bool

`prepare` receives the *noisy* circuit of the evaluation cell. Prior-based
decoders (PyMatching, Tesseract, Relay-BP) read its detector error model. A
neural decoder ignores the noise in it unless explicitly configured otherwise:
it was trained on its own training circuit, which is what makes transfer
experiments (train on uniform, test on SI1000) possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import stim


class Decoder(ABC):
    name: str = "decoder"
    device: str = "cpu"  # reported with timing results

    @abstractmethod
    def prepare(self, circuit: stim.Circuit) -> None: ...

    @abstractmethod
    def decode(self, dets: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict:
        """Settings recorded alongside every result."""
        return {"name": self.name, "device": self.device}
