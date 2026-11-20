# Reference decoders on the surface code, d = 5 and 7 (2026-11-20)

The bar Cascade has to clear (gate G1 in PLAN.md). Z-basis memory, rounds = d,
fixed test seeds (the same shots Cascade will be scored on). Logical error rate
per shot, with (failures / shots). Raw rows: `results/reference_surface.jsonl`.

| Code | Noise | p | PyMatching | Tesseract long-beam | PyMatching ÷ Tesseract |
|---|---|---|---|---|---|
| surface_d5 | uniform | 0.1% | 1.79e-03 (1073/600,000) | 1.00e-03 (20/20,000) | 1.79 |
| surface_d5 | uniform | 0.2% | 1.22e-02 (1221/100,000) | 7.95e-03 (159/20,000) | 1.54 |
| surface_d5 | uniform | 0.3% | 3.63e-02 (1816/50,000) | 2.16e-02 (216/10,000) | 1.68 |
| surface_d5 | uniform | 0.5% | 1.17e-01 (5856/50,000) | 9.44e-02 (236/2,500) | 1.24 |
| surface_d5 | si1000 | 0.1% | 1.04e-03 (1040/1,000,000) | 6.50e-04 (13/20,000) | 1.60 |
| surface_d5 | si1000 | 0.2% | 7.92e-03 (1188/150,000) | 4.50e-03 (90/20,000) | 1.76 |
| surface_d5 | si1000 | 0.3% | 2.43e-02 (1215/50,000) | 1.72e-02 (206/12,000) | 1.42 |
| surface_d5 | si1000 | 0.5% | 9.01e-02 (4507/50,000) | 6.09e-02 (213/3,500) | 1.48 |
| surface_d7 | uniform | 0.1% | 5.12e-04 (512/1,000,000) | — | — |
| surface_d7 | uniform | 0.2% | 7.46e-03 (1119/150,000) | — | — |
| surface_d7 | uniform | 0.3% | 3.15e-02 (1575/50,000) | — | — |
| surface_d7 | uniform | 0.5% | 1.43e-01 (7132/50,000) | — | — |
| surface_d7 | si1000 | 0.1% | 2.46e-04 (246/1,000,000) | — | — |
| surface_d7 | si1000 | 0.2% | 4.25e-03 (1063/250,000) | — | — |
| surface_d7 | si1000 | 0.3% | 2.00e-02 (2004/100,000) | — | — |
| surface_d7 | si1000 | 0.5% | 1.06e-01 (5312/50,000) | — | — |

Tesseract on d = 7 needs a many-core machine (it's ~seconds per shot); to do.

**What it shows**

- **Tesseract makes 1.2–1.8× fewer errors than PyMatching.** That leaves a real
  gap for Cascade to fill; G1 asks for "better than PyMatching, within 2× of
  Tesseract".
- **Uniform noise at p = 0.5% is above threshold here.** PyMatching's error rate
  *rises* from d = 5 (1.17e-1) to d = 7 (1.43e-1). Our uniform model puts every
  location (including idling) at strength p, which is harsher than the paper's
  noise, so the paper's training rate (0.7%) is probably above threshold too.
  The advisor warned about exactly this. The first GPU session therefore
  calibrates p_train (0.3%, 0.5%, 0.7%) instead of assuming 0.7%; see
  `docs/GPU_RUNBOOK.md`. (The d = 3 test run trained at 0.5% and worked well.)
- At p ≤ 0.3% both noise models are below threshold: d = 7 beats d = 5.
