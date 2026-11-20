# First Cascade training run: surface code d = 3 (2026-11-20)

**Purpose:** prove the network and training loop work end to end before
renting a GPU. Not a paper result: d = 3 is the smallest code, and the run is
tiny.

**Setup:** `surface_d3`, Z basis, trained on uniform noise at p = 0.5%;
H = 64, L = 3 (≈ 49k parameters); 2.5M training shots, batch 512, AdamW,
lr 1e-3, EMA; CPU (2 cores), 12 minutes, ~3,400 shots/s.

```bash
python scripts/train_cascade.py --code surface_d3 --p-train 0.005 \
    --samples 2.5e6 --batch 512 --hidden 64 --layers 3 --out runs/d3_proof
```

**Results:** 100,000 identical test shots per row; 95% Wilson intervals.

| Noise | p | Cascade | PyMatching | Tesseract (long-beam) |
|---|---|---|---|---|
| uniform | 0.5% | 7.79e-2 [7.63, 7.96] | 8.89e-2 [8.71, 9.06] | 7.83e-2 [7.67, 8.00] |
| uniform | 0.3% | 3.27e-2 [3.16, 3.38] | 3.74e-2 [3.62, 3.86] | 3.17e-2 [3.06, 3.28] |
| uniform | 0.2% | 1.49e-2 [1.42, 1.57] | 1.75e-2 [1.68, 1.84] | 1.47e-2 [1.40, 1.55] |
| SI1000 (never trained on) | 0.3% | 2.63e-2 [2.54, 2.73] | 3.00e-2 [2.89, 3.10] | 2.44e-2 [2.35, 2.54] |

(Logical error rate per shot; brackets are the interval × 1e-2.)

**What it shows:**
- Cascade makes 12–15% fewer errors than PyMatching at every noise level.
- On uniform noise it is statistically tied with Tesseract (the intervals overlap).
- On SI1000 noise, which it never saw in training, it still beats PyMatching
  and is ~8% behind Tesseract: a first hint for the transfer experiment.

**What it doesn't show:** d = 3 is easy, and the gap between decoders is small
here. The real checkpoint (G1) is d = 5 and 7 on a GPU.
