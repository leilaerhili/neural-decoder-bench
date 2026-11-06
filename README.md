# Neural decoder benchmark

A fair, side-by-side comparison of neural decoders for quantum error correction.

## Why this project

- **Decoders** are programs that fix errors in quantum computers: they read the clues the machine gives off and guess what went wrong.
- **Neural decoders** (AI-based) are among the most accurate, but each one has only been tested by its own authors, on their own codes, noise and setup.
- So **nobody knows how they really compare**, and people choosing a decoder have nothing fair to go on.
- **Cascade**, one of the best, has only been tested on two codes and under unrealistic, uniform noise.

## What the project does

- Tests two neural decoders, **Cascade** and a **GNN**, under **identical conditions**: same codes, same noise, same test cases, same training budget.
- **Codes (the "tracks"):**
  - the surface code;
  - the Gross code (its smaller cousin, [[72,12,6]], is used for development);
  - **the color code**, which is new for Cascade, to test whether it handles a code it wasn't designed for.
- **Noise (the "weather"):** uniform, and the more realistic **SI1000**.
- **Reference decoders:** Tesseract (near-optimal) and Relay-BP (classical).
- **Scored on:** accuracy, speed and training cost.
- **Generalization tests:** a new code (the color code) and new noise (uniform-trained models tested on SI1000).
- **Output:** a fair comparison, plus a reusable testing setup others can plug their own decoders into.

## Scope

- **In:** memory experiments only (the standard test of error correction).
- **Optional if time and money allow:** a small AlphaQubit-style transformer, fine-tuning experiments, and Z-basis to X-basis tests.
- **Out:** full AlphaQubit (no public code, too expensive to retrain) and logical operations (too complex for this project).
- **Budget:** about $150 of GPU time. If it runs short, cut in this order: Gross code under SI1000, then the Gross code entirely, then the largest surface code, then the largest color code.

## What the repo contains so far

- **Circuits** (`src/ndbench/circuits/`): the quantum experiments for all three codes, from Google's public [Tesseract repo](https://github.com/quantumlib/tesseract-decoder).
- **Noise models** (`src/ndbench/noise/`): uniform and SI1000. Our SI1000 matches Google's files exactly (checked by tests).
- **Test-case generator** (`src/ndbench/data/`): gives every decoder the same problems.
- **Reference decoders** (`src/ndbench/decoders/`): Tesseract, Relay-BP and PyMatching, all ready to run.
- **Scorekeeper** (`src/ndbench/eval/`): counts each decoder's mistakes and saves the results.
- **Tests** (`tests/`): 44 self-checks, all passing.
- **Placeholders for Cascade and the GNN** (`src/ndbench/models/`): described, not built yet.
- **The plan** ([`docs/PLAN.md`](docs/PLAN.md)): the detailed step-by-step roadmap. Decisions made along the way are in [`docs/decisions.md`](docs/decisions.md).

## The steps (about 16 weeks)

1. **Testing setup.** Mostly done ✅
2. **Build Cascade** and check it on the surface code. This is the hardest step.
3. **Build the GNN** and check it. Easier, since its authors shared their code.
4. **Move both to the BB codes.** Measure the cost first; use the smaller code if the Gross code is too expensive.
5. **Adapt both to the color code.**
6. **Test under realistic SI1000 noise.**
7. **Measure speed and training cost.**
8. **Write up and share.**

## Next steps

- Email the Cascade authors asking for their code.
- Ask about university computing time.
- Start step 2: building Cascade.

## How to run it

Needs Python 3.11 or 3.12 (one of the tools doesn't work with 3.13 yet). On a Mac, in Terminal:

```bash
# one-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh        # installs uv, which sets up Python
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[baselines,dev]"

# each time you open a new Terminal window
source .venv/bin/activate

# run the self-checks (should say 44 passed)
pytest
```

Run a decoder on one test setting. This example gives PyMatching 20,000 problems on the medium surface code with realistic noise:

```bash
python scripts/eval_reference.py --decoder pymatching --code surface_d5 \
    --noise si1000 --p 0.003 --max-shots 20000
```

The score is printed and saved to `results/`.

## Credits

Circuits come from [quantumlib/tesseract-decoder](https://github.com/quantumlib/tesseract-decoder) (Apache-2.0); see `NOTICE`.
