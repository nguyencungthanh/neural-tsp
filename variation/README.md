# Variation: Distribution-Shift (Out-of-Distribution Generalization)

This folder is a **self-contained extension** to the main project. It does not
modify any existing code. It reuses the parent project's trained policy, neural
decoders, and C++ baselines, and adds one new experiment:

> *Does the learned TSP policy generalize to point distributions it was not trained on?*

The project trains and evaluates exclusively on **uniform-random** TSP-20. That
leaves the central critique of neural combinatorial optimization untested: a
model that looks competitive on uniform data may have learned uniform-specific
shortcuts rather than general tour-building skill. Classical heuristics
(Nearest-Neighbor, 2-Opt) have **no training distribution** — they are algorithms
that work on any point set — so they are the natural yardstick for
distribution-robustness.

## What it does

Generate four evaluation distributions (all `n=20`, coords in `[0,1]^2`):

| Distribution | Geometry | Why |
|---|---|---|
| `uniform` | i.i.d. `U[0,1]^2` | **in-distribution control** (matches training) |
| `clustered` | 5 Gaussian blobs | inter-blob hub-and-spoke tour structure, never seen |
| `grid` | jittered lattice | highly regular; optimal tour is a snake |
| `ring` | noisy circle | 1-D manifold; optimal tour goes around |

Run the same methods as the main project on each, then compare. Because absolute
tour length is not comparable across distributions (clustered points sit closer
together, so *every* method scores lower there), the meaningful cross-distribution
quantity is the **gap to 2-Opt**: `(neural - 2-Opt) / 2-Opt`. If the neural policy's
gap *widens* out-of-distribution while 2-Opt holds, that is the signal that the
policy learned uniform-specific structure.

## Files

| File | Purpose |
|---|---|
| `generate_ood.py` | Generate + self-validate the four datasets (fixed seeds, reproducible) |
| `eval_neural_ood.py` | Run greedy + sampling on each dataset with a PointerNet policy |
| `run_study.py` | **One-command orchestrator**: data → baselines → neural → gap table |
| `data/` | Generated datasets (`uniform.txt`, `clustered.txt`, `grid.txt`, `ring.txt`) |
| `_build/` | Compiled baseline binary (auto-created) |

## Run

```bash
# From the repo root.

# Full study (intended for Colab, where python/actor.pt lives):
python variation/run_study.py                      # 1000 instances/distribution, N=1280

# Quick local CPU check (baselines are exact + fast; neural capped for speed):
python variation/run_study.py --instances 100 --samples 1280

# Baselines only (always works on any machine; the real reference numbers):
python variation/run_study.py --no-neural

# Regenerate data alone:
python variation/generate_ood.py --num 1000 --n 20 --seed 0
```

## Model weights

The neural columns require the project's RL policy `actor.pt`, which lives on
Colab. `run_study.py` resolves weights in this order: explicit `--model` →
`actor.pt` → `model.pt`, skipping any whose architecture does not match the
current `model.py`. If no compatible weights are found it prints the
baselines-only table (which contains the real reference numbers and is
meaningful on its own).

## How to read the result

- **`uniform` row** anchors the in-distribution baseline: here the neural policy
  should be ≈ 2-Opt (the project's headline finding).
- **`clustered` / `grid` / `ring` rows** test transfer. A growing positive gap
  column means the policy degrades where the data departs from uniform.
- 2-Opt's row is the distribution-agnostic reference; if it stays flat while the
  neural gap grows, the study has surfaced a generalization limitation — an
  honest, reportable finding that does not require the neural method to "win."

This variation does not need to improve any number; its contribution is the
*characterization* of where the approach holds and where it breaks.
