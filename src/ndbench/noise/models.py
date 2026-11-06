"""Circuit-level noise models applied to a noiseless Stim circuit.

Both models are applied moment by moment (a moment is the span between TICKs)
to the *same* noiseless circuit, so the only thing that changes between a
uniform and an SI1000 experiment is the noise.

SI1000 (Gidney, Newman & McEwen 2022, "superconducting-inspired"), as used in
the tesseract-decoder testdata circuits. The test suite checks that this
implementation reproduces those files' detector error models exactly.

    1-qubit Clifford          DEPOLARIZE1(p/10) after
    2-qubit Clifford          DEPOLARIZE2(p) after
    idle (any moment)         DEPOLARIZE1(p/10)   (only after the qubit's first operation)
    idle during M or R moment additionally DEPOLARIZE1(2p)
    reset                     flip 2p after
    measurement               flip 5p, then DEPOLARIZE1(p) on the measured qubit
                              (surface-code references only; see si1000())

Uniform depolarizing ("SD6"-style; every location has strength p):

    1-qubit Clifford          DEPOLARIZE1(p) after
    2-qubit Clifford          DEPOLARIZE2(p) after
    idle (any moment)         DEPOLARIZE1(p)
    reset                     flip p after
    measurement               flip p
"""

from __future__ import annotations

from dataclasses import dataclass

import stim

ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"}
RESETS = {"R": "X_ERROR", "RZ": "X_ERROR", "RX": "Z_ERROR", "RY": "X_ERROR"}
MEASURES = {"M", "MZ", "MX", "MY"}


@dataclass(frozen=True)
class NoiseModel:
    name: str
    p: float
    clifford1: float
    clifford2: float
    idle: float
    idle_extra_during_mr: float
    reset_flip: float
    measure_flip: float
    post_measure_depol: float

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        return _apply(self, circuit.without_noise().flattened())


def si1000(p: float, post_measure_depol: bool = True) -> NoiseModel:
    """SI1000. The reference surface-code circuits add DEPOLARIZE1(p) after each
    measurement; the reference BB circuits do not. `library.noisy` picks the
    convention matching each code family so both reproduce their references.
    """
    return NoiseModel("si1000", p, clifford1=p / 10, clifford2=p, idle=p / 10,
                      idle_extra_during_mr=2 * p, reset_flip=2 * p, measure_flip=5 * p,
                      post_measure_depol=p if post_measure_depol else 0.0)


def uniform(p: float, **_) -> NoiseModel:
    return NoiseModel("uniform", p, clifford1=p, clifford2=p, idle=p,
                      idle_extra_during_mr=0.0, reset_flip=p, measure_flip=p,
                      post_measure_depol=0.0)


MODELS = {"si1000": si1000, "uniform": uniform}


def get(name: str, p: float, **kw) -> NoiseModel:
    return MODELS[name](p, **kw)


def _moments(circuit: stim.Circuit):
    """Split a flat circuit into lists of instructions separated by TICK."""
    cur: list[stim.CircuitInstruction] = []
    for inst in circuit:
        if inst.name == "TICK":
            yield cur, True
            cur = []
        else:
            cur.append(inst)
    yield cur, False


def _qubits(inst: stim.CircuitInstruction) -> list[int]:
    return [t.value for t in inst.targets_copy() if t.is_qubit_target]


def _apply(nm: NoiseModel, circuit: stim.Circuit) -> stim.Circuit:
    """Insert noise after each operation, in order, then idle noise per moment.

    Idle qubits = qubits that have had an operation before (`active`) and are
    not acted on in this moment. In a moment with several measurement
    instructions (e.g. ancillas then data at the end of a BB memory
    experiment), qubits not yet touched idle right after the first one, which
    is how the reference BB circuits are written.
    """
    all_qubits = range(circuit.num_qubits)
    active: set[int] = set()
    out = stim.Circuit()

    def idle_noise(qubits: list[int], during_mr: bool) -> None:
        if not qubits:
            return
        out.append("DEPOLARIZE1", qubits, nm.idle)
        if during_mr and nm.idle_extra_during_mr > 0:
            out.append("DEPOLARIZE1", qubits, nm.idle_extra_during_mr)

    for moment, ends_with_tick in _moments(circuit):
        has_mr = any(i.name in MEASURES or i.name in RESETS for i in moment)
        touched: set[int] = set()
        idled: set[int] = set()
        seen_measure = False
        for inst in moment:
            name = inst.name
            if name in ANNOTATIONS:
                out.append(inst)
                continue
            qs = _qubits(inst)
            touched.update(qs)
            if name in MEASURES:
                out.append(name, inst.targets_copy(), nm.measure_flip)
                if nm.post_measure_depol:
                    out.append("DEPOLARIZE1", qs, nm.post_measure_depol)
                if not seen_measure:
                    seen_measure = True
                    waiting = [q for q in all_qubits if q in active and q not in touched]
                    idle_noise(waiting, True)
                    idled.update(waiting)
                continue
            out.append(inst)
            gd = stim.gate_data(name)
            if name in RESETS:
                if nm.reset_flip:
                    out.append(RESETS[name], qs, nm.reset_flip)
            elif gd.is_two_qubit_gate:
                out.append("DEPOLARIZE2", qs, nm.clifford2)
            elif gd.is_unitary:
                out.append("DEPOLARIZE1", qs, nm.clifford1)
            else:
                raise NotImplementedError(f"no noise rule for {name}")
        idle_noise([q for q in all_qubits if q in active and q not in touched and q not in idled],
                   has_mr)
        active.update(touched)
        if ends_with_tick:
            out.append("TICK")
    return out
