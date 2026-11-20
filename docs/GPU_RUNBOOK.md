# GPU runbook: training Cascade on a rented GPU

Step-by-step instructions for the first GPU session (Phase 2: Cascade on the
surface code, distances 5 and 7). Read it once fully before renting: the meter
runs from the moment the machine starts.

## 0. Before renting (free, on your laptop)

- `pytest` passes on your laptop, and everything is pushed to GitHub (the GPU
  machine downloads the code from there).
- Check whether the university can give you GPU time (free). If yes, skip
  step 1 and run steps 2–7 on the cluster instead.

## 1. Rent the machine

Any GPU rental service works (RunPod, Lambda, Vast.ai, ...). What to pick:

| Setting | Choose | Why |
|---|---|---|
| GPU | One consumer or mid-range card (e.g. RTX 4090, L4, A10, A100). Not 8 GPUs. | Our models are small (≈ 0.3–1M parameters); a big GPU barely helps. Run the speed test (step 3) and compare price per run, not price per hour. |
| CPUs | 8 or more | Workers generate training problems on the CPU. |
| Disk | 30–50 GB | PyTorch alone is several GB. |
| Image/template | One with PyTorch + CUDA preinstalled | Saves setup time. |
| On-demand vs spot | On-demand for the first session | Spot is cheaper but can stop at any moment. Training resumes automatically, so spot is fine once things work. |

Prices change; check them on the day. Rough guide: consumer GPUs well under
$1/hour, A100/H100 a few $/hour.

## 2. Set up (≈ 10 minutes)

Open the machine's terminal (web terminal or `ssh`), then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/leilaerhili/neural-decoder-bench
cd neural-decoder-bench
git checkout cascade-surface          # or main, once merged
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[baselines,nn,dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
pytest -q
```

The `python -c` line must print `True` and the GPU's name. If it prints
`False`, PyTorch can't see the GPU: stop and ask before going further.

## 3. Speed test (≈ 5 minutes)

```bash
python scripts/benchmark_training.py --price-per-hour 0.70   # use your machine's real price
```

It prints samples/s, hours and cost per 1e8-sample run for distances 5 and 7.
Pick the batch size and number of workers with the highest samples/s. Write
the numbers into `docs/results/` (they're the "G0c-surface pilot" in the plan).

**Stop and rethink if** a distance-7 run would cost more than about $15:
the whole core project has to fit in about $150.

## 4. Train (runs for hours: keep it alive with tmux)

`tmux` keeps the run going if your browser or laptop disconnects.

```bash
tmux new -s train
source .venv/bin/activate

# First: 5 short tuning runs (1e7 samples each), on distance 5.
# (a) Which training noise level? Uniform p = 0.5% is already above threshold
#     on our circuits (docs/results/2026-11-20_reference_surface.md), so the
#     paper's 0.7% may be too high.
for PT in 0.003 0.005 0.007; do
  python scripts/train_cascade.py --code surface_d5 --p-train $PT --samples 1e7 \
      --batch 4096 --hidden 128 --layers 5 --lr 1e-3 --workers 4 --curriculum 0.02 \
      --out runs/tune_d5_p${PT}
done
```

The validation sets differ between those runs (each is at its own p), so
compare them on the *same* test shots instead:

```bash
for PT in 0.003 0.005 0.007; do
  python scripts/eval_checkpoint.py runs/tune_d5_p${PT}/final.pt --noise uniform --p 0.003 --shots 200000
done
```

Pick the p_train with the fewest errors, then two learning-rate runs at it:

```bash
PT=0.005   # <- the winner from above
for LR in 3e-4 3e-3; do
  python scripts/train_cascade.py --code surface_d5 --p-train $PT --samples 1e7 \
      --batch 4096 --hidden 128 --layers 5 --lr $LR --workers 4 --curriculum 0.02 \
      --out runs/tune_d5_p${PT}_lr${LR}
done
```

Detach with `Ctrl-b` then `d`. Reattach later with `tmux attach -t train`.
Watch progress from another terminal:

```bash
tail -f runs/*/log.jsonl
```

Compare the learning rates the same way (all three now share p_train, so `val_ler_ema`
in each `log.jsonl` works too). Then the real runs, with the winning p_train and LR:

```bash
LR=1e-3   # <- the winning learning rate (1e-3 is the run from the noise-level step)
for SEED in 0 1 2; do
  python scripts/train_cascade.py --code surface_d5 --p-train $PT --samples 1e8 \
      --batch 4096 --hidden 128 --layers 5 --lr $LR --workers 4 --curriculum 0.02 \
      --seed $SEED --out runs/cascade_d5_uniform_s$SEED
done
python scripts/train_cascade.py --code surface_d7 --p-train $PT --samples 1e8 \
    --batch 4096 --hidden 128 --layers 7 --lr $LR --workers 4 --curriculum 0.02 \
    --out runs/cascade_d7_uniform_s0
```

**If the machine stops** (spot instance, crash): start it again, `cd` into the
folder, activate, and rerun the *same* command. It resumes from
`runs/<name>/latest.pt` and produces exactly the same model as an
uninterrupted run.

## 5. Score the models (on the GPU machine)

```bash
for P in 0.001 0.002 0.003 0.005; do
  python scripts/eval_checkpoint.py runs/cascade_d5_uniform_s0/final.pt --p $P --shots 1000000
done
```

This scores Cascade and PyMatching on the same shots and prints the ratio.
Tesseract comparisons use `results/reference_surface.jsonl`, computed on a CPU.

## 6. Bring the results home

The trained models are not in git (too big). Copy them to your laptop. From
your **laptop's** terminal (the host and port are on the rental page):

```bash
scp -P <port> -r root@<host>:neural-decoder-bench/runs ~/code/neural-decoder-bench/
scp -P <port> root@<host>:neural-decoder-bench/results/neural.jsonl ~/code/neural-decoder-bench/results/
```

Then commit `results/neural.jsonl` and a short write-up in `docs/results/`.

## 7. Stop the machine

**Stop or terminate it on the rental website.** Closing the browser tab does
not stop billing. Check the billing page afterwards.

## What "success" looks like (gate G1 in PLAN.md)

On distances 5 and 7, uniform noise, p ≤ 0.3%:
- Cascade makes fewer errors than PyMatching, and
- at most 2× Tesseract's errors, and
- going from d = 5 to d = 7 reduces errors at least as much as it does for PyMatching.
