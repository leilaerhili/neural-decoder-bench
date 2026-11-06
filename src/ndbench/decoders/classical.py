"""Reference decoders: PyMatching (surface-code sanity check), Tesseract, Relay-BP.

Imports are lazy so the core package works without the `baselines` extra.
"""

from __future__ import annotations

import numpy as np
import stim

from .base import Decoder


class PyMatchingDecoder(Decoder):
    name = "pymatching"

    def prepare(self, circuit: stim.Circuit) -> None:
        import pymatching

        dem = circuit.detector_error_model(decompose_errors=True)
        self._m = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, dets: np.ndarray) -> np.ndarray:
        return self._m.decode_batch(dets).astype(bool)


class _SinterWrapped(Decoder):
    """Adapter for any sinter.Decoder: compile for the DEM, decode bit-packed shots."""

    def _sinter_decoder(self):
        raise NotImplementedError

    def prepare(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model()
        self._num_obs = dem.num_observables
        self._compiled = self._sinter_decoder().compile_decoder_for_dem(dem=dem)

    def decode(self, dets: np.ndarray) -> np.ndarray:
        packed = np.packbits(dets, axis=1, bitorder="little")
        out = self._compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        return np.unpackbits(out, axis=1, bitorder="little", count=self._num_obs).astype(bool)


class TesseractDecoder(_SinterWrapped):
    """Near-optimal search decoder (Beni, Higgott & Shutty 2025).

    `preset` is one of tesseract-decoder's sinter presets; "tesseract-short-beam"
    and "tesseract-long-beam" are the two settings used in the Tesseract paper.
    Long-beam is the accuracy reference; short-beam is cheaper for sweeps.
    """

    name = "tesseract"

    def __init__(self, preset: str = "tesseract-long-beam"):
        self.preset = preset

    def _sinter_decoder(self):
        from tesseract_decoder import make_tesseract_sinter_decoders_dict

        return make_tesseract_sinter_decoders_dict()[self.preset]

    def describe(self) -> dict:
        return {**super().describe(), "preset": self.preset}


class RelayBPDecoder(_SinterWrapped):
    """Relay-BP (Müller et al. 2025) through its sinter integration.

    Defaults are the settings in the relay-bp package README for the Gross code.
    """

    name = "relay_bp"

    DEFAULTS = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                    gamma_dist_interval=(-0.24, 0.66), stop_nconv=1)

    def __init__(self, **kwargs):
        self.kwargs = {**self.DEFAULTS, **kwargs}

    def _sinter_decoder(self):
        from relay_bp.stim import sinter_decoders

        return sinter_decoders(**self.kwargs)["relay-bp"]

    def describe(self) -> dict:
        return {**super().describe(), **{k: str(v) for k, v in self.kwargs.items()}}


REGISTRY = {
    "pymatching": PyMatchingDecoder,
    "tesseract": TesseractDecoder,
    "relay_bp": RelayBPDecoder,
}
