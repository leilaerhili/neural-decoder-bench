"""Cascade (Gu, Bonilla Ataides, Lukin & Yelin 2026, arXiv:2604.08358). Phase 2 / 4.

Reimplementation target, from the paper (no public code):

- Embedding: binary detection events -> H-dim vectors.
- Backbone: L bottleneck residual blocks (H -> H/4 -> conv -> H), each
  projection/conv preceded by BatchNorm + SiLU, residual branch scaled 1/sqrt(2L).
- Surface code: syndrome tensor (R, d+1, d+1), 3x3x3 conv, zero padding, L ~ d.
- BB codes: torus-equivariant conv factored check -> data -> check, each step
  over the 6 Tanner neighbours x 2 time offsets; 1D conv over adjacent rounds.
  The paper uses custom Triton kernels; here a gather + einsum over index
  tables built from the check matrices (slower, same maths).
- Color code (our extension; not in the paper): the same gather-based
  convolution over the hexagonal lattice's neighbour tables, zero-padded at
  the triangle's boundary, as on the surface code.
- Readout: conv scatters check features to data qubits, average-pool over each
  logical observable's support, 2-layer MLP (width 2H), one logit per observable.
- Training (paper): BCE; Muon (conv weights, lr 3e-3) + Lion (rest, lr 2e-4);
  wd 3e-3; EMA 0.9998; cosine schedule with 1k warmup; batch 3328; bf16;
  short low-p curriculum (<2% of steps) then train at p_train (0.7% surface,
  0.55% BB, uniform noise).

Our budget forces smaller models (H 64-128, L 6-8) and fewer samples; the
reproduction gate (G1) is relative to Tesseract/PyMatching on our circuits.
"""

from __future__ import annotations


class Cascade:  # placeholder until Phase 2
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Cascade is implemented in Phase 2 (see docs/PLAN.md)")
