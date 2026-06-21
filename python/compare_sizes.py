"""
Size comparison: Active Search (neural) vs the exact optimum (Held-Karp) and the
classical heuristics (NN, 2-Opt), across n in {10, 12, 15}.

Why Active Search only: the user wants the strongest neural inference strategy,
not greedy/sampling. Active Search is per-instance iterative -- it runs
`NUM_STEPS` gradient *steps* on each instance -- so it is FAR more expensive than
a forward pass. 1000 instances would take ~tens of hours even on a T4; that is
why we use a small N (default 50). Held-Karp / NN / 2-Opt are evaluated on the
SAME instances so the mean +/- std comparison stays apples-to-apples.

Prerequisites (the notebook does these before %run-ning this script):
  - actor.pt exists (the n=15 RL policy) in python/
  - ../data/comp_n{10,12,15}.txt      raw instances (points only)
  - ../data/comp_opt_n{10,12,15}.txt  Held-Karp optimal tours (make_supervised output)
  - cpp/baselines compiled (this script compiles it if stale)

Run (cwd = python/):
    python compare_sizes.py
or in Colab:
    %run compare_sizes.py
"""
import os
import io
import sys
import contextlib
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PointerNet  # noqa: E402  (ensures class is importable for active_search)
from dataset import TSPDataset            # points + tour (optimal tours)  # noqa: E402
from dataset_rl import TSPDatasetRL       # points only (raw instances)    # noqa: E402
from eval_utils import tour_length        # noqa: E402
from active_search import active_search   # noqa: E402

# ============================ knobs (edit me) ============================
SIZES = [10, 12, 15]
NUM_INSTANCES = 50        # active search is expensive; 1000 is infeasible (~tens of h)
NUM_STEPS = 5000          # gradient steps per instance (matches eval_rl.py / paper)
BATCH_SIZE = 128
AS_LR = 1e-5              # active-search learning rate (pretrained start)
ACTOR_PATH = "actor.pt"
DATA_DIR = "../data"
CPP_DIR = "../cpp"
OUT_PNG = "../data/size_comparison.png"
OUT_CSV = "../data/size_comparison.csv"
# =========================================================================

METHODS = [
    ("optimal", "Held-Karp (optimal)", "black"),
    ("nn", "Nearest Neighbor", "tab:orange"),
    ("2opt", "2-Opt", "tab:green"),
    ("active_search", "Active Search (neural)", "tab:blue"),
]


def ensure_baselines():
    binp = os.path.join(CPP_DIR, "baselines")
    src = os.path.join(CPP_DIR, "baselines.cpp")
    if not os.path.exists(binp) or os.path.getmtime(src) > os.path.getmtime(binp):
        print("[baselines] compiling cpp/baselines.cpp ...")
        subprocess.run(["g++", "-std=c++17", "-O2", src, "-o", binp], check=True)
    return binp


def run_baselines_per_instance(binp, data_path):
    """Run NN + 2-Opt in C++ with --per-instance; return (nn[], 2opt[])."""
    with open(data_path) as fh:
        out = subprocess.run([binp, "--per-instance"], stdin=fh,
                             capture_output=True, text=True, check=True)
    nn, two = [], []
    for line in out.stdout.splitlines():
        if line.startswith("PI "):
            _, _i, a, b = line.split()
            nn.append(float(a))
            two.append(float(b))
    return np.array(nn), np.array(two)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    assert os.path.exists(ACTOR_PATH), (
        f"{ACTOR_PATH} not found -- train the n=15 model (train_rl.py) first.")

    binp = ensure_baselines()

    results = {key: {} for key, _, _ in METHODS}

    for n in SIZES:
        raw_path = os.path.join(DATA_DIR, f"comp_n{n}.txt")
        opt_path = os.path.join(DATA_DIR, f"comp_opt_n{n}.txt")
        for p in (raw_path, opt_path):
            assert os.path.exists(p), (
                f"missing {p} -- generate comp data and run make_supervised first.")

        # --- optimal lengths (from Held-Karp tours emitted by make_supervised) ---
        ds_opt = TSPDataset(opt_path)
        opt_lens = np.array([
            tour_length(pts.unsqueeze(0).to(device), tour.unsqueeze(0).to(device)).item()
            for pts, tour in ds_opt
        ])

        # --- NN / 2-Opt (C++, per instance) ---
        nn_lens, two_lens = run_baselines_per_instance(binp, raw_path)

        # --- Active Search (neural), per instance ---
        # Reload actor.pt fresh each instance: active search MUTATES the weights,
        # so every instance must start from the original pretrained policy.
        ds_raw = TSPDatasetRL(raw_path)
        n_inst = min(NUM_INSTANCES, len(ds_raw), len(opt_lens), len(nn_lens))
        print(f"\n[n={n}] Active Search on {n_inst} instances, {NUM_STEPS} steps each "
              f"(C++ baselines already done) ...")
        as_lens = []
        for i in range(n_inst):
            # suppress active_search's per-1000-step chatter; print our own 1 line
            with contextlib.redirect_stdout(io.StringIO()):
                _, best = active_search(
                    ds_raw[i], state_dict_path=ACTOR_PATH,
                    num_steps=NUM_STEPS, batch_size=BATCH_SIZE,
                    lr=AS_LR, device=device)
            as_lens.append(best)
            print(f"  instance {i+1}/{n_inst}: active_search length = {best:.4f}")

        # slice every method to the same instances (fair mean/std)
        results["optimal"][n] = opt_lens[:n_inst]
        results["nn"][n] = nn_lens[:n_inst]
        results["2opt"][n] = two_lens[:n_inst]
        results["active_search"][n] = np.array(as_lens)

    n_inst = min(NUM_INSTANCES, min(len(v) for v in results["optimal"].values()))

    # ---------------- summary table ----------------
    print("\n" + "=" * 72)
    print(f"mean +/- std tour length per method  (N={n_inst} instances/size)")
    print("=" * 72)
    header = f"{'method':<22}" + "".join(f"{'n='+str(n):>16}" for n in SIZES)
    print(header)
    print("-" * 72)
    for key, label, _ in METHODS:
        cells = []
        for n in SIZES:
            arr = results[key][n]
            cells.append(f"{arr.mean():>7.4f}+/-{arr.std():.4f}")
        print(f"{label:<22}" + "".join(f"{c:>16}" for c in cells))

    # ---------------- CSV ----------------
    import csv
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] +
                   [f"n{n}_mean" for n in SIZES] +
                   [f"n{n}_std" for n in SIZES])
        for key, label, _ in METHODS:
            means = [results[key][n].mean() for n in SIZES]
            stds = [results[key][n].std() for n in SIZES]
            w.writerow([label] + means + stds)
        # per-instance optimality gaps too
        w.writerow([])
        w.writerow(["gap_vs_optimal_%"] + [f"n{n}_mean" for n in SIZES] + [f"n{n}_std" for n in SIZES])
        for key, label, _ in METHODS:
            if key == "optimal":
                continue
            gmean, gstd = [], []
            for n in SIZES:
                g = (results[key][n] - results["optimal"][n]) / results["optimal"][n] * 100
                gmean.append(g.mean())
                gstd.append(g.std())
            w.writerow([label] + gmean + gstd)
    print(f"\nSaved table -> {OUT_CSV}")

    # ---------------- plot ----------------
    xs = np.array(SIZES, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) absolute length vs n, mean +/- std bands
    for key, label, color in METHODS:
        means = np.array([results[key][n].mean() for n in SIZES])
        stds = np.array([results[key][n].std() for n in SIZES])
        ax1.plot(xs, means, "o-", color=color, label=label, linewidth=2)
        ax1.fill_between(xs, means - stds, means + stds, color=color, alpha=0.15)
    ax1.set_xlabel("n (cities)")
    ax1.set_ylabel("average tour length")
    ax1.set_title(f"Tour length vs n  (mean +/- std, N={n_inst} inst/size)")
    ax1.set_xticks(SIZES)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # (b) optimality gap vs n
    for key, label, color in METHODS:
        if key == "optimal":
            continue
        gmean = np.array([((results[key][n] - results["optimal"][n]) /
                           results["optimal"][n] * 100).mean() for n in SIZES])
        gstd = np.array([((results[key][n] - results["optimal"][n]) /
                          results["optimal"][n] * 100).std() for n in SIZES])
        ax2.plot(xs, gmean, "o-", color=color, label=label, linewidth=2)
        ax2.fill_between(xs, gmean - gstd, gmean + gstd, color=color, alpha=0.15)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("n (cities)")
    ax2.set_ylabel("optimality gap  (method - optimal)/optimal  [%]")
    ax2.set_title("Optimality gap vs n  (lower = better)")
    ax2.set_xticks(SIZES)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Neural (Active Search) vs Held-Karp & heuristics across n", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
