"""Use a trained neural decoder through the same interface as every other decoder.

    dec = NeuralDecoder("runs/cascade_d5/final.pt")
    result = evaluate(dec, "surface_d5", "si1000", 0.003)

`prepare` checks that the evaluation circuit has the same layout the model was
trained on, but ignores its noise: a neural decoder only knows the noise it
was trained on. That's what makes transfer tests (train on uniform, test on
SI1000) possible.
"""

from __future__ import annotations

import numpy as np
import stim
import torch

from .base import Decoder


class NeuralDecoder(Decoder):
    name = "neural"

    def __init__(self, checkpoint: str, device: str = "auto", batch: int = 4096):
        from ..training import load_checkpoint, pick_device

        self.checkpoint = str(checkpoint)
        self._device = pick_device(device)
        self.device = str(self._device)
        self.batch = batch
        self.model, ck = load_checkpoint(checkpoint, device=str(self._device))
        self.train_config = ck["train_config"]
        self.samples = ck["samples"]
        self.name = "cascade"

    def prepare(self, circuit: stim.Circuit) -> None:
        from ..models.layout import SurfaceLayout

        self.layout = SurfaceLayout.from_circuit(circuit)
        trained_shape = tuple(self.model.type_channels.shape[1:])
        if self.layout.shape[1:] != trained_shape:
            raise ValueError(f"circuit grid {self.layout.shape[1:]} != trained grid {trained_shape}")

    @torch.no_grad()
    def decode(self, dets: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(dets), self.batch):
            grid = torch.from_numpy(self.layout.to_tensor(dets[i:i + self.batch])).to(self._device)
            out.append((self.model(grid) > 0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.support.shape[0]), bool)

    def describe(self) -> dict:
        tc = self.train_config
        return {**super().describe(), "checkpoint": self.checkpoint, "trained_samples": self.samples,
                "train_noise": tc["noise"], "train_p": tc["p_train"], "model": tc["model"]}
