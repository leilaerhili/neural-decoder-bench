"""Checks for the Cascade network, training loop and decoder wrapper.

Skipped automatically if PyTorch isn't installed (`pip install -e ".[nn]"`).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ndbench.circuits import noiseless, noisy  # noqa: E402
from ndbench.data import Sampler  # noqa: E402
from ndbench.decoders.neural import NeuralDecoder  # noqa: E402
from ndbench.eval.runner import evaluate  # noqa: E402
from ndbench.models.cascade import CascadeConfig, CascadeSurface  # noqa: E402
from ndbench.models.layout import SurfaceLayout  # noqa: E402
from ndbench.training import TrainConfig, load_checkpoint, train  # noqa: E402

SMALL = {"hidden": 16, "layers": 2}


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("code", ["surface_d3", "surface_d5"])
def test_observable_support(code, basis):
    """The logical observable is one full row (Z basis) or column (X basis) of data qubits."""
    d = int(code[-1])
    lay = SurfaceLayout.from_circuit(noiseless(code, basis))
    s = lay.support[0]
    assert s.shape == (d, d) and s.sum() == d
    line_sums = s.sum(axis=1) if basis == "Z" else s.sum(axis=0)
    assert sorted(line_sums.tolist()) == [0] * (d - 1) + [d]


@pytest.mark.parametrize("code", ["surface_d3", "surface_d5", "surface_d7"])
def test_forward_shapes(code):
    lay = SurfaceLayout.from_circuit(noiseless(code, "Z"))
    model = CascadeSurface(lay, CascadeConfig(**SMALL)).eval()
    dets = Sampler(noisy(code, "uniform", 0.005), 1).sample(8).dets
    out = model(torch.from_numpy(lay.to_tensor(dets)))
    assert out.shape == (8, 1) and torch.isfinite(out).all()


def test_gradients_reach_every_parameter():
    lay = SurfaceLayout.from_circuit(noiseless("surface_d3", "Z"))
    model = CascadeSurface(lay, CascadeConfig(**SMALL))
    shots = Sampler(noisy("surface_d3", "uniform", 0.01), 2).sample(64)
    logits = model(torch.from_numpy(lay.to_tensor(shots.dets)))
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.from_numpy(shots.obs.astype(np.float32))).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_no_detections_gives_same_answer_everywhere_in_batch():
    """Sanity: identical inputs give identical outputs (eval mode, no randomness)."""
    lay = SurfaceLayout.from_circuit(noiseless("surface_d5", "Z"))
    model = CascadeSurface(lay, CascadeConfig(**SMALL)).eval()
    out = model(torch.zeros(4, *lay.shape))
    assert torch.allclose(out, out[:1].expand_as(out))


def test_parameter_count_scales_with_size():
    lay = SurfaceLayout.from_circuit(noiseless("surface_d5", "Z"))
    small = CascadeSurface(lay, CascadeConfig(hidden=64, layers=5)).num_parameters()
    big = CascadeSurface(lay, CascadeConfig(hidden=128, layers=5)).num_parameters()
    assert 3 < big / small < 4.5  # roughly quadratic in hidden size


def test_train_checkpoint_and_decode(tmp_path):
    """A tiny run end to end: train, save, reload, decode through the runner."""
    cfg = TrainConfig(code="surface_d3", noise="uniform", p_train=0.005, model=SMALL,
                      samples=4096, batch=512, warmup_steps=2, log_every=4, val_shots=1000,
                      device="cpu", checkpoints_at=[2048])
    final = train(cfg, tmp_path, quiet=True)
    assert final.exists() and (tmp_path / "ckpt_2e+03.pt").exists()
    assert (tmp_path / "log.jsonl").read_text().count("\n") >= 2

    model, ck = load_checkpoint(final)
    assert ck["samples"] == 4096 and ck["train_config"]["code"] == "surface_d3"

    dec = NeuralDecoder(str(final), device="cpu")
    r = evaluate(dec, "surface_d3", "si1000", 0.003, max_shots=2000, max_failures=10**9, batch=1000)
    assert r.shots == 2000 and r.decoder["train_noise"] == "uniform"


def test_wrong_code_is_rejected(tmp_path):
    cfg = TrainConfig(code="surface_d3", model=SMALL, samples=512, batch=512, val_shots=100,
                      device="cpu", warmup_steps=1, log_every=1)
    dec = NeuralDecoder(str(train(cfg, tmp_path, quiet=True)), device="cpu")
    with pytest.raises(ValueError):
        dec.prepare(noisy("surface_d5", "uniform", 0.003))


@pytest.mark.slow
def test_learns_to_beat_trivial(tmp_path):
    """After ~200k shots on d = 3 the model clearly beats 'never predict a flip'."""
    cfg = TrainConfig(code="surface_d3", noise="uniform", p_train=0.005,
                      model={"hidden": 32, "layers": 3}, samples=200_000, batch=512,
                      warmup_steps=20, log_every=100, val_shots=5000, device="cpu")
    dec = NeuralDecoder(str(train(cfg, tmp_path, quiet=True)), device="cpu")
    shots = Sampler(noisy("surface_d3", "uniform", 0.005), 99).sample(20_000)
    dec.prepare(noisy("surface_d3", "uniform", 0.005))
    nn_err = (dec.decode(shots.dets) != shots.obs).any(axis=1).mean()
    trivial_err = shots.obs.any(axis=1).mean()
    assert nn_err < 0.6 * trivial_err


# --- GPU-readiness: deterministic shot stream, parallel workers, resume -----

from ndbench.training import block_p, block_seed, ordered_batches  # noqa: E402
from ndbench.data import split_of  # noqa: E402


def _tiny_cfg(**kw):
    base = dict(code="surface_d3", noise="uniform", p_train=0.005, model=SMALL, samples=512 * 12,
                batch=512, block_steps=3, warmup_steps=2, log_every=4, save_every=5,
                val_shots=500, device="cpu", checkpoints_at=[])
    base.update(kw)
    return TrainConfig(**base)


def _batches(cfg, start=0):
    lay = SurfaceLayout.from_circuit(noiseless(cfg.code, cfg.basis))
    return [g.numpy().copy() for g, _ in ordered_batches(cfg, lay, start)]


def test_shot_stream_same_for_any_number_of_workers():
    serial = _batches(_tiny_cfg(workers=0))
    parallel = _batches(_tiny_cfg(workers=2))
    assert len(serial) == len(parallel) == 12
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)


def test_shot_stream_resumes_mid_block():
    full = _batches(_tiny_cfg())
    tail = _batches(_tiny_cfg(), start=7)  # step 7 is in the middle of block 2
    for a, b in zip(full[7:], tail):
        np.testing.assert_array_equal(a, b)


def test_training_seeds_stay_in_train_range_and_curriculum_ramps():
    cfg = _tiny_cfg(samples=512 * 1000, curriculum_frac=0.1)
    assert all(split_of(block_seed(cfg, b)) == "train" for b in range(0, 400))
    ps = [block_p(cfg, b) for b in range(0, 60)]
    assert ps[0] == pytest.approx(0.25 * 0.005) and ps[-1] == pytest.approx(0.005)
    assert all(x <= y for x, y in zip(ps, ps[1:]))
    assert block_p(_tiny_cfg(), 0) == 0.005  # no curriculum by default


def test_interrupted_training_resumes_to_identical_result(tmp_path):
    cfg = _tiny_cfg()
    uninterrupted = load_checkpoint(train(cfg, tmp_path / "a", quiet=True))[1]["weights"]
    stopped = train(cfg, tmp_path / "b", quiet=True, stop_after_steps=7)
    assert stopped.name == "latest.pt"
    resumed = load_checkpoint(train(cfg, tmp_path / "b", quiet=True))[1]["weights"]
    assert not (tmp_path / "b" / "latest.pt").exists()
    for k in uninterrupted:
        assert torch.allclose(uninterrupted[k].float(), resumed[k].float(), atol=1e-6), k


def test_resume_refuses_a_different_config(tmp_path):
    train(_tiny_cfg(), tmp_path, quiet=True, stop_after_steps=6)
    with pytest.raises(ValueError):
        train(_tiny_cfg(lr=5e-4), tmp_path, quiet=True)


def test_auto_device_never_picks_apple_gpu(monkeypatch):
    """Apple's MPS backend crashes on Cascade's 3D convolutions on some macOS
    versions, so "auto" must fall back to the CPU when there's no NVIDIA GPU."""
    from ndbench.training import pick_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("auto").type == "cpu"
    assert pick_device("cpu").type == "cpu"
