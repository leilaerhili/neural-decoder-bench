# Implementation plan

**Project.** A controlled benchmark of neural decoders for QEC. Cascade and a
GNN decoder are trained and evaluated under identical conditions on
memory experiments: surface code, Gross code and color code, uniform
depolarizing and SI1000 noise, with Tesseract and Relay-BP as references.
Each decoder is scored on accuracy, decoding speed and training cost, plus
two generalization tests: a **new code** (Cascade has only been shown on
surface and BB codes; the color code is new for it) and a **new noise model**
(uniform-trained models evaluated on SI1000).

**Deliverables.** (1) Open benchmark harness; (2) results for the core grid;
(3) paper/report with practical guidance on decoder choice.

---

## 1. Design decisions

| ID | Decision | Why |
|---|---|---|
| D1 | **Circuits:** use the public tesseract-decoder circuits (rotated surface code with CZ gates; BB codes with a colored CZ schedule; triangular color code with the superdense cycle of Gidney & Jones 2023), stored noiseless; rounds = distance; Z-basis memory. | Public, used in the Tesseract paper, cover all three code families. Fixing the circuit means only the decoder and the noise vary. Checked: every BB CZ layer applies whole polynomial terms uniformly around the torus, so the circuit is translation-invariant and Cascade's weight sharing fits it. |
| D2 | **Noise:** apply our own noise models to the noiseless circuits. SI1000 reproduces the reference files' detector error models exactly (tested). Uniform depolarizing = every location at strength p (SD6-style). | Same circuit under both noise models; transfer experiments become clean. |
| D3 | **Codes:** surface d = 5, 7 (d = 3 for debugging); color d = 5, 7 (d = 3 for debugging); BB [[72,12,6]] for development; Gross [[144,12,12]] for final results, gated by the BB compute pilot (G0c-BB, Phase 4). | The color code is the generalization test the advisor suggested: Cascade has never been shown on it, its circuits are public, and small distances are cheap to train. Gross is the stated target, but a full-size Cascade on it costs up to ~100 H200-hours in the paper; [[72,12,6]] has 4× fewer detectors and is where bugs get found. |
| D4 | **Model sizes:** Cascade H = 128, L = 6 to 8 (paper's ablation: H ≤ 64 fails to extrapolate). GNN as published (~1.4M parameters). | Largest size that fits the budget, above the known failure size. |
| D5 | **Training error rate:** calibrated per (model, code, noise) with the same procedure for every model (section 5). Uniform starting points from the Cascade paper (0.7% surface, 0.55% BB); the color code has no published value, so it gets the full candidate sweep. | The advisor flagged that SI1000 accumulates faster and the paper's rate may not train. |
| D6 | **Fairness:** same circuits, same shots at evaluation, same training-sample budget, same tuning budget (6 short runs on `surface_d5` uniform, plus 2 per model for each new code), learning curves for all models, no early stopping (the final EMA checkpoint is the model; validation data is used only for tuning and p_train calibration). | Fairness is the paper's main claim to value. |
| D7 | **GNN on BB codes:** surface code uses the published kNN graph on detector coordinates; BB codes use edges from the detector error model (detectors that share an error mechanism). Ablate both graph types on the surface code. | Euclidean distance between BB detector coordinates doesn't reflect torus adjacency; the ablation shows the adaptation isn't a handicap. |
| D8 | **Cascade convolution on non-square codes:** one gather + einsum layer over neighbour index tables, used for both the BB torus and the color code's hexagonal lattice (built from each code's checks and detector coordinates; missing neighbours at the color code's boundary are zero-padded, as Cascade does on the surface code). No custom Triton kernels. | One mechanism covers both new geometries, so the color-code adaptation is a faithful extension rather than a redesign. Correctness first; speed is reported as "our implementation" alongside FLOP counts. |
| D9 | **Reference decoders:** Relay-BP (package's Gross-code settings) on every cell; Tesseract with the paper's `long-beam` preset as the near-optimal anchor at selected cells on BB codes and on every surface- and color-code cell. Settings recorded with every result. | Tesseract's tutorial settings fail on BB codes (measured: LER 0.47 vs Relay-BP's 0.07 on [[72,12,6]], SI1000 p = 0.3%), so presets are used; Tesseract is ~100–1000× slower than Relay-BP on BB codes (section 6), so it can only anchor, not sweep. |
| D10 | **Reproducibility:** every result row records the git commit, package versions, decoder settings and seed; dependencies locked with `uv lock`; checkpoints saved with their config. | Lets anyone rerun a single cell. |

---

## 2. Phases

Times assume ~12–15 hours a week, about 16 weeks in total including 2 weeks
of buffer. The critical path is Phase 2 → Phases 4 and 4b (Cascade on the BB and color
codes); the GNN work runs alongside it. Each phase ends at a checkpoint (G); a failed checkpoint
triggers its fallback, not a stop.

**Week 1 actions outside the code:** email the Cascade authors asking for code
or checkpoints; ask about university GPU/CPU allocations (the single biggest
lever on scope); set up a GPU rental account with checkpoint-to-storage so
spot instances can be used.

### Phase 1: Harness and references (weeks 1–2)

Done in the skeleton:
- [x] Noiseless circuits vendored; uniform and SI1000 noise models
- [x] SI1000 reproduces 12 reference circuits exactly (surface, BB and color codes; test suite)
- [x] Seeded sampling with disjoint train/val/test seed ranges
- [x] Decoder interface; PyMatching, Tesseract, Relay-BP wrappers
- [x] Evaluation runner (paired shots, Wilson intervals, per-round and per-logical-qubit LER, JSONL results)

To do:
1. Parallel evaluation for CPU decoders (multiprocessing over batches, or sinter directly).
2. CPU cost table: seconds per shot for Tesseract (short- and long-beam) and Relay-BP on every core cell, from ~100 shots each. Then fix the reference sweep: shot caps and p grid per decoder so the whole sweep fits the CPU budget.
3. Reference sweep (D9): Relay-BP on every cell; Tesseract long-beam on every surface- and color-code cell and at the BB anchor cells in `configs/grid.toml`; PyMatching on the surface code.
4. Compare with published numbers where the circuit is the same: Tesseract paper results on these exact circuits.
5. Analysis scaffolding: `analysis/` scripts that turn result JSONL into the figures in section 3, run on the reference results first.

**G0a** SI1000 reproduces references ✅.
**G0b** References behave as expected: Tesseract-long-beam ≤ Relay-BP at the BB anchor cells and ≤ PyMatching on the surface code (within CI), and Tesseract matches its paper where comparable. *Fallback:* tune presets; if Tesseract is too slow on Gross, use short-beam and report it as such.

### Phase 2: Cascade on the surface code (weeks 2–5), critical path

1. Detector layout: map each detector to (round, row, col) from its coordinates; unit test that the mapping is a bijection onto the (R, d+1, d+1) grid's occupied cells.
2. Model per `ndbench/models/cascade.py`; readout pools over the logical observable's support.
3. Training loop: on-the-fly Stim sampling (worker processes), BCE, Muon + Lion (fallback AdamW if Muon misbehaves), cosine schedule, EMA, bf16, low-p curriculum; checkpoints at 1e6, 3e6, 1e7, 3e7, 1e8 samples.
4. **Surface pilot (G0c-surface):** rent a GPU for ~2 hours; measure training throughput (samples/s) for Cascade at H = 128 and the Stim sampling rate on that machine's CPUs. If sampling can't keep up, pre-generate shards on a CPU box.
5. Tuning: 6 short runs (1e7 samples): learning rate × {H = 64, 128}.
6. Train `surface_d5`, uniform, p_train = 0.7%, 3 seeds; then `surface_d7`, 1 seed.

**G1 (Cascade reproduction).** Our circuits differ from the paper's, so the criteria are relative: on `surface_d5` and `surface_d7` uniform at p ≤ 0.3%, Cascade beats PyMatching and is within 2× of Tesseract-long-beam (the paper puts full-size Cascade between correlated matching and Tesseract; ours is smaller and trained on less data); error suppression from d = 5 to 7 is at least PyMatching's.
*Fallbacks, in order:* (a) debug list (input mapping, readout support, label basis, learning rate, train p); (b) email the authors for code or checkpoints (do this in week 1 regardless); (c) report as a "Cascade-style CNN" with the measured gap stated.

### Phase 3: GNN on the surface code (weeks 3–5), runs alongside Phase 2

1. Validate the reference code: load the published pretrained weights and evaluate them on the circuits they were trained on (Stim-generated surface code); must match the paper's reported accuracy within CI.
2. Port the model into `ndbench` (PyTorch Geometric), with graph construction on the GPU (`torch_cluster.knn`) so it doesn't bottleneck training or inflate latency; train on our `surface_d5` uniform circuit with the same tuning budget as Cascade.

**G1'** Pretrained weights reproduce the paper; our retrained model is within 1.5× of the pretrained one's LER on a common circuit.

### Phase 4: BB codes (weeks 5–9)

1. BB detector → (cell, check type, round) map from the circuit, checked against `ndbench.codes.bb` (torus translations must permute detectors).
2. Cascade torus convolution; unit test: translating the input syndrome on the torus translates the internal features (equivariance).
3. GNN DEM-graph builder; surface-code ablation (kNN vs DEM graph).
4. **BB pilot (G0c-BB):** time forward + backward passes of both models on `bb72` and `bb144` (random weights, minutes of GPU time); project the remaining grid. This decides whether Gross is in (section 6 cut order).
5. Two short tuning runs per model on `bb72`; then train both on `bb72` uniform; then `bb144` uniform if G0c-BB passed.

**G2** On `bb72` uniform, Cascade beats Relay-BP at p ≤ 0.3% (the paper reports a large margin). *Fallback:* as G1.

### Phase 4b: Color code, the new-code generalization test (weeks 7–9, alongside Phase 4)

1. Color detector → (round, lattice site, check colour) map from detector coordinates; neighbour tables for the hexagonal lattice (reusing the D8 gather convolution); unit test that each detector's neighbours match the checks it shares qubits with.
2. GNN: no adaptation needed (detector coordinates are planar, so the published kNN graph applies).
3. Two short tuning runs per model; train both on `color_d5` and `color_d7`, uniform.
4. Report against Tesseract (near-optimal and fast enough here) and Relay-BP. BP-based decoders are known to struggle on color codes, so Tesseract is the meaningful bar.

**G2c** No pass/fail threshold: whatever Cascade achieves on a code it wasn't designed for *is* the result. Sanity check only: it must beat a trivial decoder and improve from d = 5 to d = 7 at low p; if not, debug before reporting.

### Phase 5: SI1000 and transfer (weeks 9–11)

1. Calibrate p_train for SI1000 (section 5) for each model and code.
2. Train SI1000 models (surface d5, d7; color d5, d7; bb72; bb144 if budget).
3. Transfer test (no training): evaluate uniform-trained models on SI1000 cells with the same test shots as the SI1000-trained ones. The quantity reported is the paired LER ratio (uniform-trained ÷ SI1000-trained) at each p: the cost of training on the wrong noise model. Comparing "uniform at p" with "SI1000 at p" directly is avoided, since equal p does not mean equal noise strength.

### Phase 6: Speed and cost (week 11)

- Inference: per-shot latency at batch 1 and throughput at the best batch size, same GPU for all neural models; CPU single-thread for Tesseract and Relay-BP. Timing is end to end from detection events (includes graph building for the GNN and tensor layout for Cascade). Report hardware, precision, batch size. GPU and CPU numbers are not compared as if equivalent, and no real-time claims are made: the paper's 1 µs target needs dedicated hardware.
- Training cost: GPU-hours and samples to reach within 1.2× of the final LER (from learning curves); parameters; FLOPs per shot.

### Phase 7: Optional extensions (week 12, only after the core grid)

In order:
1. Small compute-matched transformer (covers the third architecture family).
2. Cascade ablations: which parts matter (e.g. the bottleneck blocks, the factored convolution, the readout pooling)? The advisor named this as the question he'd explore with more time.
3. Fine-tuning uniform → SI1000.
4. Z → X basis transfer.

### Phase 8: Write-up and release (weeks 13–16, including buffer)

Figures, tables, paper draft; tagged release of code, configs, checkpoints
and result files.

---

## 3. Evaluation protocol

- **Metric:** logical error rate per round, and per round per logical qubit for BB codes (12 observables), with 95% Wilson intervals. A shot fails if any observable is wrong.
- **Shots:** sample until 100 failures or a shot cap. Caps: neural 1e6, Relay-BP 1e5, Tesseract 1e4 (surface) and 1e3–1e4 at its anchor cells (BB). Paired comparisons use a fixed shot count from the same test seed.
- **Measurable range:** with 1e4–1e5 shots, the reference decoders resolve LER/shot down to ~1e-3–1e-4; neural decoders on GPU resolve ~1e-5. Neural-vs-reference comparisons are made at higher p; neural-vs-neural extends to lower p. No extrapolation below measured points.
- **Statistics:** McNemar on paired shots; bootstrap LER ratios; seed variance from 3 seeds on the development config (`surface_d5` uniform).
- **Planned figures and tables:**
  1. LER vs p for every decoder, one panel per (code, noise).
  2. Learning curves: LER vs training samples and vs GPU-hours, per model and code.
  3. Transfer: uniform-trained ÷ SI1000-trained LER ratio vs p.
  3b. New code: each decoder's LER relative to Tesseract on the color code vs on the surface code (does Cascade's edge carry over to a code it wasn't designed for?).
  4. Accuracy vs inference latency vs training cost (one point per decoder per code).
  5. Summary table: best decoder per (code, noise, compute constraint), with the margin and its CI.

## 4. Core grid

| | surface_d5 | surface_d7 | color_d5 | color_d7 | bb72 | bb144 |
|---|---|---|---|---|---|---|
| uniform | C, G | C, G | C, G | C, G | C, G | C, G (gated) |
| SI1000 | C, G | C, G | C, G | C, G | C, G | C, G (gated) |

C = Cascade, G = GNN. 24 training runs at 1e8 samples (the 8 color-code runs
are small: 90–252 detectors), plus ~35 short runs (tuning, calibration; 1e7
samples each) and 4 seed repeats. Every cell is also evaluated with Relay-BP;
Tesseract on every surface- and color-code cell and at the BB anchor cells;
PyMatching on surface codes.

## 5. Training-rate calibration (same for every model)

For each (model, code, noise): candidates p_train ∈ {0.2, 0.3, 0.4, 0.5}% for
SI1000 and for uniform noise on the color code (no published value); for
uniform noise on surface and BB codes, the paper's value ± one step. Short run (1e7 samples) per
candidate; choose the one with the lowest validation LER at the middle of
the evaluation grid. Ties go to the higher p. Logged as part of the result.

## 6. Compute budget

**GPU (~$150, roughly 50–70 H100-hours).** Estimate per full run =
1e8 samples ÷ training throughput. Back-of-envelope: the paper's largest BB
model trained 3e8 samples in < 100 H200-hours (~800 samples/s). Our Cascade is
~30× fewer FLOPs per sample (H 128 vs 512, L 8 vs 14), so roughly
10–20k samples/s ideal and plausibly 5–10k/s without custom kernels: ~3–6
GPU-hours per Gross run, ~1–2 per [[72,12,6]] run, minutes for surface codes.
**These are estimates; the pilots in Phases 2 and 4 replace them.**

**CPU (separate, cheap per hour but large).** Measured single-threaded on a
2-core container, [[72,12,6]], SI1000:

| Decoder | p = 0.1% | p = 0.3% |
|---|---|---|
| Relay-BP | 0.02 s/shot | 0.1 s/shot |
| Tesseract short-beam | 0.4 s/shot | ~18 s/shot |
| Tesseract long-beam | 0.7 s/shot | slower (not measured) |
| PyMatching (surface d = 5) | ~5 µs/shot | ~5 µs/shot |

Color code d = 5, SI1000, p = 0.2%: Tesseract long-beam ~0.1 s/shot, Relay-BP
~6 ms/shot (and less accurate: 10 vs 3 failures in 300 shots). d = 7 is
several times slower for Tesseract.

Cost grows steeply with p and code size, so the roles are split:

- **Relay-BP** is the full-sweep classical reference (every cell, up to 1e5 shots).
- **Tesseract** is the near-optimal *anchor*: on BB codes, long-beam at 2 p values per code with 1e3–1e4 shots; on the surface and color codes, every cell (it's fast enough there, and on the color code it is the meaningful bar because BP-based decoders struggle).
- Rough total: a few hundred CPU-hours for [[72,12,6]] and a few thousand for Gross at these settings. That's affordable on a university cluster or a spot many-core VM (a 64-core VM turns 50 CPU-hours into under an hour), but it rules out Tesseract on every Gross cell.

**Cut order if the pilot says the grid doesn't fit:** (1) Gross SI1000
runs; (2) Gross entirely, keeping [[72,12,6]] as the BB code; (3) surface d = 7;
(4) color d = 7. The color code is kept ahead of the Gross code because it is
the new-code generalization test and is cheap. Optional extensions are
dropped before any of these.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Cascade underperforms the paper | Relative gate G1; contact authors in week 1; report the gap. |
| Reference decoders misconfigured | Paper presets; G0b sanity checks; settings logged with results. |
| Compute | Pilot before the grid; cut order; tuning on cheap surface-code configs. |
| GNN adaptation questioned on BB | Ablation D7; clearly labeled as an adaptation. |
| Cascade's color-code version is our design, not the authors' | Built from the same mechanism as the BB version (D8); described as an extension; a poor result is reported with that caveat, not as a verdict on Cascade itself. |
| Scoop (Cascade authors or others) | Release the harness and an early note once G1/G2 pass. |

---

## 8. Review log (what changed after critiquing the first draft)

1. **Pilot moved to where it can run.** The first draft had the compute pilot in Phase 1, before any model exists. It's now split: surface throughput in Phase 2, BB throughput in Phase 4 (random-weight timing), with the Gross go/no-go decided there.
2. **Tesseract's role changed.** Measured costs (section 6) make Tesseract on every BB cell impossible. Relay-BP is the full-sweep reference; Tesseract long-beam anchors selected cells. Tutorial settings were also found to truncate the search on BB codes, so paper presets are used.
3. **G1 loosened from 1.5× to 2× of Tesseract**, since our Cascade is smaller and trained on less data than the paper's; error-suppression and beat-PyMatching criteria stay.
4. **Fairness tightened:** tuning budget defined per new code, no early stopping, graph construction timed as part of GNN latency, GNN graphs built on the GPU.
5. **Transfer metric made precise:** the paired ratio of uniform-trained to SI1000-trained LER on the same SI1000 shots, not a comparison across noise models at "equal p".
6. **Reproducibility rule added (D10)** and implemented: result rows carry the git commit and package versions.
7. **Translation invariance of the BB circuit checked** before committing to Cascade's equivariant convolution on it.
8. **Timeline:** 2 weeks of buffer added; week-1 actions (email the Cascade authors, ask about cluster time) made explicit.
9. **Color code added to the core project (after re-reading the advisor's notes).** He suggested adapting Cascade to other public circuits, such as the color code, to see how well it generalizes; the first plan only used the two codes Cascade was already shown on. The color code is now in the core grid (d = 5, 7), uses the same circuits source and noise models (SI1000 reproduces its references exactly), and is kept ahead of the Gross code in the cut order.
10. **Advisor's other suggestion recorded:** Cascade ablations (which parts matter) added as optional extension 2.

Remaining known weaknesses: the GPU estimates are back-of-envelope until the pilots; two neural decoders is a thin comparison (mitigated by the optional transformer and the open harness); the CPU timings come from a slow 2-core container and will be faster on normal hardware; the color-code version of Cascade is our own extension, so its result speaks to that adaptation as much as to Cascade.
