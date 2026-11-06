"""GNN decoder (Lange et al., Phys. Rev. Research 7, 023181 (2025)). Phase 3 / 4.

Reference implementation: github.com/LangeMoritz/GNN_decoder (MIT), surface
code only, with pretrained circuit-level weights for d = 3..9.

- Nodes: fired detectors only; features = one-hot check type + (x, y, t).
- Edges (surface code, as published): k nearest fired detectors (k = 10),
  edge weight = inverse squared Euclidean distance.
- Edges (BB codes, our adaptation): Euclidean distance on a torus layout is
  misleading, so nodes are joined when they share an error mechanism in the
  detector error model (or are within m hops of it). Ablated on the surface
  code against the published kNN graph (see docs/PLAN.md, D7).
- 7 GraphConv layers (32 -> 512 -> 256), mean pooling, MLP head
  (256-128-64), one sigmoid output per observable; Adam, lr 1e-4 -> 1e-5.
"""

from __future__ import annotations


class GNNDecoderModel:  # placeholder until Phase 3
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GNN is implemented in Phase 3 (see docs/PLAN.md)")
