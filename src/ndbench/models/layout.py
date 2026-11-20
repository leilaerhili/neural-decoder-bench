"""Turn a surface-code shot into the 3D picture Cascade looks at.

A shot is a flat list of detection events (one True/False per detector). Cascade
is a convolutional network, so it needs those events arranged spatially: a
stack of (d+1) x (d+1) images, one per round, where each pixel is one
stabilizer at one point in time. Neighbouring pixels are neighbouring
stabilizers, which is what lets a convolution spot local error patterns.

The tesseract-decoder circuits give every detector coordinates (x, y, t, k):
  x, y in {-0.5, 0.5, ..., d - 0.5}  -> pixel (row, col) = (y + 0.5, x + 0.5)
  t    in {0, ..., d}                 -> time slice
  k    in {0, 1} for X-type checks, {3, 4} for Z-type checks

Not every pixel is a stabilizer (the rotated surface code leaves corners and
alternating edge cells empty), and the first and last time slices only contain
one check type. Empty pixels are zero, as in the Cascade paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

X_TYPE, Z_TYPE = 1, 2  # values in `cell_type`; 0 = no stabilizer at that pixel


@dataclass(frozen=True)
class SurfaceLayout:
    shape: tuple[int, int, int]  # (T, H, W)
    t: np.ndarray  # (D,) time slice of each detector
    row: np.ndarray  # (D,)
    col: np.ndarray  # (D,)
    det_type: np.ndarray  # (D,) X_TYPE or Z_TYPE
    cell_type: np.ndarray  # (H, W) which check lives at each pixel (0 = none)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> "SurfaceLayout":
        coords = circuit.get_detector_coordinates()
        n = circuit.num_detectors
        if len(coords) != n or any(len(coords[i]) < 4 for i in range(n)):
            raise ValueError("every detector needs (x, y, t, k) coordinates")
        c = np.array([coords[i][:4] for i in range(n)], dtype=float)
        col = np.rint(c[:, 0] + 0.5).astype(int)
        row = np.rint(c[:, 1] + 0.5).astype(int)
        t = np.rint(c[:, 2]).astype(int)
        det_type = np.where(c[:, 3] >= 2, Z_TYPE, X_TYPE)
        if (col < 0).any() or (row < 0).any() or (t < 0).any():
            raise ValueError("unexpected negative grid index")
        shape = (int(t.max()) + 1, int(row.max()) + 1, int(col.max()) + 1)

        if len(set(zip(t, row, col))) != n:
            raise ValueError("two detectors map to the same pixel")
        cell_type = np.zeros(shape[1:], dtype=np.int8)
        for r, cc, k in zip(row, col, det_type):
            if cell_type[r, cc] not in (0, k):
                raise ValueError(f"pixel ({r}, {cc}) holds both check types")
            cell_type[r, cc] = k
        return cls(shape, t, row, col, det_type, cell_type)

    def to_tensor(self, dets: np.ndarray) -> np.ndarray:
        """(shots, D) bool -> (shots, T, H, W) float32, empty pixels = 0."""
        dets = np.asarray(dets)
        out = np.zeros((len(dets), *self.shape), dtype=np.float32)
        out[:, self.t, self.row, self.col] = dets
        return out

    def from_tensor(self, grid: np.ndarray) -> np.ndarray:
        """Inverse of `to_tensor` (reads the detector pixels back out)."""
        return np.asarray(grid)[:, self.t, self.row, self.col].astype(bool)

    def type_channels(self) -> np.ndarray:
        """(2, H, W) float32 masks: is this pixel an X check, a Z check?

        A convolution applies the same rule everywhere, so it can't tell which
        pixels are X or Z checks unless that's part of its input. These two
        fixed channels are fed alongside the detection events.
        """
        return np.stack([self.cell_type == X_TYPE, self.cell_type == Z_TYPE]).astype(np.float32)
