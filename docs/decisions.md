# Decision log

Short, dated entries: the choice, why, and what would reverse it. The main
design decisions are D1–D10 in `PLAN.md`; this file records changes and
findings along the way, newest last.

## 2026-11-03: Python 3.11–3.12 only
`sinter` (needed by the Tesseract and Relay-BP integrations) does not build on
Python 3.13 yet, so `pyproject.toml` requires `>=3.11,<3.13` and the setup
instructions use 3.12. Reverse when sinter ships 3.13 wheels.

## 2026-11-03: Circuits from tesseract-decoder, noise applied by us
All circuits come from the public tesseract-decoder testdata and are stored
noiseless; our noise models are applied on top. This keeps the circuit fixed
across noise models (uniform vs SI1000), so only the noise changes.
Reverse if: a circuit family we need isn't there (then generate it with Stim
and document the difference).

## 2026-11-03: SI1000 conventions differ between the reference circuit families
The tesseract-decoder surface-code circuits add `DEPOLARIZE1(p)` after every
measurement; the BB-code circuits don't, and they apply the data qubits' idle
noise before the final data readout. `noise.si1000(p, post_measure_depol=...)`
plus in-order noise insertion reproduces both families exactly;
`circuits.library.noise_model` picks the convention by code family.
Reverse if: we switch to a single SI1000 convention for all codes (then
results are no longer on the published circuits).

## 2026-11-03: Tesseract settings
The tutorial configuration (`pqlimit=10000`, 1 detector order) truncates the
search on BB codes: LER/shot 0.47 on [[72,12,6]], SI1000 p = 0.3%, versus 0.07
for Relay-BP. The wrapper therefore uses the paper's `tesseract-long-beam`
(default) and `tesseract-short-beam` presets.

## 2026-11-03: Tesseract anchors, Relay-BP sweeps
Measured single-threaded on [[72,12,6]], SI1000: Tesseract short-beam takes
~0.4 s/shot at p = 0.1% and ~18 s/shot at p = 0.3%; Relay-BP takes 0.02–0.1
s/shot. Running Tesseract on every BB cell is therefore out of reach, so
Relay-BP is the reference on every cell and Tesseract (long-beam) anchors a few
cells per BB code (`configs/grid.toml`). On the surface and color codes
Tesseract is fast enough to run everywhere.
Reverse if: we get enough CPU time (e.g. a cluster allocation) to run
Tesseract on every cell.

## 2026-11-03: BB circuit is translation-invariant
Every CZ layer of the BB circuits applies whole polynomial terms uniformly
around the torus (gate offsets constant modulo the lattice), so the noisy
circuit is invariant under torus translations and Cascade's weight sharing fits it.
To be turned into a unit test in Phase 4.

## 2026-11-03: Plan revised after critique
The first draft of `PLAN.md` was reviewed and revised (its section 8 lists the
changes): pilots moved to where the models exist, G1 loosened to 2× of
Tesseract, fairness and transfer metric made precise, reproducibility rule
(D10) added, buffer added to the timeline.

## 2026-11-04: Color code added to the core grid
Re-reading the advisor's notes: he suggested taking Cascade to other public
circuits (the color code) to test how it generalizes, since it has only been
shown on surface and BB codes. Added the triangular color code (superdense
cycle, d = 3, 5, 7) from the same tesseract-decoder testdata. SI1000 reproduces
its references exactly with either measurement convention; we use the
surface-code convention. The color code sits ahead of the Gross code in the
cut order. Early timing (d = 5, SI1000, p = 0.2%): Tesseract ~0.1 s/shot and
clearly more accurate than Relay-BP (3 vs 10 failures in 300 shots), so
Tesseract is the meaningful reference on this code.
Reverse if: adapting Cascade to the hexagonal lattice turns out to need a
redesign rather than the D8 neighbour-table convolution.

## 2026-11-06: Cascade ablations added as an optional extension
The advisor said that with more time he would study which parts of Cascade
matter. Added as optional extension 2 in `PLAN.md` (after the small
transformer), not in the core grid, because the core grid already uses the
budget.

## 2026-11-06: README rewritten as a plain-language overview
The README now explains the project, scope, repo contents and next steps in
plain language, plus setup instructions. Technical detail lives in `PLAN.md`.
