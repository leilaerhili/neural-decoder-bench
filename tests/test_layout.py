"""Checks that the surface-code picture Cascade sees is correct."""

import numpy as np
import pytest

from ndbench.circuits import noiseless, noisy
from ndbench.data import Sampler
from ndbench.models.layout import X_TYPE, Z_TYPE, SurfaceLayout

CODES = ["surface_d3", "surface_d5", "surface_d7"]


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_grid_shape_and_stabilizer_count(code, basis):
    d = int(code[-1])
    lay = SurfaceLayout.from_circuit(noiseless(code, basis))
    assert lay.shape == (d + 1, d + 1, d + 1)  # T = rounds + final readout
    # A distance-d rotated surface code has d^2 - 1 stabilizers, half X and half Z.
    assert (lay.cell_type == X_TYPE).sum() == (d * d - 1) // 2
    assert (lay.cell_type == Z_TYPE).sum() == (d * d - 1) // 2


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_first_and_last_slices_hold_only_the_memory_basis(code, basis):
    lay = SurfaceLayout.from_circuit(noiseless(code, basis))
    want = Z_TYPE if basis == "Z" else X_TYPE
    for t in (0, lay.shape[0] - 1):
        assert set(lay.det_type[lay.t == t]) == {want}


@pytest.mark.parametrize("code", CODES)
def test_round_trip(code):
    c = noisy(code, "si1000", 0.003)
    lay = SurfaceLayout.from_circuit(c)
    dets = Sampler(c, 7).sample(200).dets
    grid = lay.to_tensor(dets)
    np.testing.assert_array_equal(lay.from_tensor(grid), dets)
    # Nothing ever lands on an empty pixel.
    assert grid[:, :, lay.cell_type == 0].sum() == 0
    assert grid.sum() == dets.sum()


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("noise", ["uniform", "si1000"])
def test_every_error_is_local_in_the_picture(code, noise):
    """The premise of a convolutional decoder: any single physical error lights
    up detectors that are close together in the grid. Each error in the
    detector error model must fit inside a 3 x 3 x 3 window (Cascade's kernel)."""
    c = noisy(code, noise, 0.003)
    lay = SurfaceLayout.from_circuit(c)
    worst = 0
    for inst in c.detector_error_model(decompose_errors=False).flattened():
        if inst.type != "error":
            continue
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        if len(ds) < 2:
            continue
        ds = np.array(ds)
        span = max(np.ptp(lay.t[ds]), np.ptp(lay.row[ds]), np.ptp(lay.col[ds]))
        worst = max(worst, span)
    assert worst <= 2, f"an error spans {worst + 1} pixels in some direction"


def test_type_channels():
    lay = SurfaceLayout.from_circuit(noiseless("surface_d5", "Z"))
    ch = lay.type_channels()
    assert ch.shape == (2, 6, 6)
    assert ch.sum() == 24 and (ch.sum(axis=0) <= 1).all()
