"""
Distribution-shift study: orchestrates data generation, classical baselines
(NN, 2-Opt), and neural decoding (greedy + sampling) across four point
distributions, then prints a gap-to-2-Opt comparison table.

This is the single entry point for the variation. It reuses the parent
project's C++ baselines and neural model/search code -- nothing is duplicated.

Why "gap to 2-Opt": absolute tour length is NOT comparable across distributions
(clustered points are closer together, so all methods score lower there). The
scientifically meaningful cross-distribution quantity is how the neural policy
*stands relative to* 2-Opt on each distribution. 2-Opt is distribution-agnostic
local search (no training set), so a widening neural-vs-2Opt gap out-of-distribution
is the signal that the policy learned uniform-specific structure rather than
general TSP-solving.

Local run (CPU; will use model.pt only if architecture matches, else skips neural):
  python variation/run_study.py --instances 200 --samples 1280

Full Colab run (with the trained actor.pt):
  python variation/run_study.py                     # 1000 instances, N=1280

Baselines only (always works, fast):
  python variation/run_study.py --no-neural
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # generate_ood, eval_neural_ood
sys.path.insert(0, os.path.join(HERE, "..", "python"))  # parent model/search

import subprocess
import numpy as np  # noqa: E402  (used by generate_ood)

import generate_ood as G
from eval_neural_ood import load_policy, evaluate

REPO = os.path.abspath(os.path.join(HERE, ".."))
CPP_BASELINES = os.path.join(REPO, "cpp", "baselines.cpp")
BUILD_DIR = os.path.join(HERE, "_build")
DATA_DIR = os.path.join(HERE, "data")
DISTRIBUTIONS = ["uniform", "clustered", "grid", "ring"]

# torch is optional at this layer (baselines work without it).
try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    _DEVICE = None


def ensure_data(num, n, seed):
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {}
    for name in DISTRIBUTIONS:
        p = os.path.join(DATA_DIR, f"{name}.txt")
        if not os.path.exists(p):
            print(f"[data] generating {name} ({num} x {n}) ...")
            pts = G.generate_distribution(name, num, n, seed=seed)
            G.write_tsp_file(p, pts)
            G.validate(p, num, n)  # read-back check
        paths[name] = p
    return paths


def ensure_baselines():
    os.makedirs(BUILD_DIR, exist_ok=True)
    binp = os.path.join(BUILD_DIR, "baselines")
    # Recompile if missing, stale, OR not executable. The last check matters because a
    # binary copied from Google Drive loses its +x bit, so an existing-but-non-executable
    # binary would raise PermissionError when subprocess tries to run it.
    if (not os.path.exists(binp)
            or os.path.getmtime(CPP_BASELINES) > os.path.getmtime(binp)
            or not os.access(binp, os.X_OK)):
        print("[baselines] compiling cpp/baselines.cpp ...")
        subprocess.run(["g++", "-std=c++17", "-O2", CPP_BASELINES, "-o", binp], check=True)
    return binp


def run_baselines(binp, data_path):
    """Run NN + 2-Opt on one file; return (nn_avg, twoopt_avg)."""
    with open(data_path) as fh:
        out = subprocess.run([binp], stdin=fh, capture_output=True, text=True, check=True)
    nn = two = None
    for line in out.stdout.splitlines():
        low = line.lower()
        if low.startswith("nn's"):
            nn = float(line.split(":")[-1].strip())
        elif low.startswith("2opt's"):
            two = float(line.split(":")[-1].strip())
    if nn is None or two is None:
        raise RuntimeError(f"could not parse baseline output:\n{out.stdout}")
    return nn, two


def main():
    ap = argparse.ArgumentParser(description="Distribution-shift study (OOD generalization).")
    ap.add_argument("--num", type=int, default=1000, help="instances per distribution (generation)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1280, help="neural sampling N")
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--instances", type=int, default=None,
                    help="cap neural eval to first N instances (use small N for a CPU check)")
    ap.add_argument("--no-neural", action="store_true", help="skip neural decoding (baselines only)")
    ap.add_argument("--model", default=None, help="explicit policy weights path")
    args = ap.parse_args()

    # --- 1. data ---
    paths = ensure_data(args.num, args.n, args.seed)

    # --- 2. baselines (always) ---
    binp = ensure_baselines()
    print("\n[baselines] running NN + 2-Opt on each distribution ...")
    baseline = {}
    for name in DISTRIBUTIONS:
        nn, two = run_baselines(binp, paths[name])
        baseline[name] = {"nn": nn, "2opt": two}
        print(f"  {name:<11} NN={nn:.4f}  2-Opt={two:.4f}")

    # --- 3. neural (optional) ---
    neural = {}
    neural_src = None
    if not args.no_neural:
        if torch is None:
            print("\n[neural] torch unavailable -- skipping (baselines-only run).")
        else:
            model, neural_src = load_policy(args.model, device=_DEVICE)
            if model is None:
                print("\n[neural] no compatible policy weights found (need actor.pt on Colab). "
                      "Skipping neural columns.")
            else:
                print(f"\n[neural] policy={neural_src} device={_DEVICE} "
                      f"samples={args.samples} T={args.temperature}")
                if neural_src == "model.pt":
                    print("  (model.pt = supervised stand-in; NOT the project's RL actor.pt)")
                model.eval()
                for name in DISTRIBUTIONS:
                    print(f"  decoding {name} ...")
                    res = evaluate(paths[name], model, device=_DEVICE,
                                   num_instances=args.instances,
                                   num_samples=args.samples,
                                   temperature=args.temperature, verbose=False)
                    neural[name] = res
                    print(f"    {name:<11} greedy={res['greedy']:.4f}  "
                          f"sampling={res['sampling']:.4f}  (N_inst={res['n']})")

    # --- 4. table ---
    print("\n" + "=" * 78)
    title = "Distribution-shift study: tour length (lower=better) and gap to 2-Opt"
    print(title)
    print("=" * 78)
    has_neural = bool(neural)
    if has_neural:
        hdr = f"{'dist':<11}{'NN':>9}{'2-Opt':>9}{'Neur-Gr':>9}{'Neur-Sa':>9}{'Gr-2opt':>10}{'Sa-2opt':>10}"
        print(hdr)
        print("-" * 78)
        for name in DISTRIBUTIONS:
            b = baseline[name]
            g = neural[name]["greedy"]
            s = neural[name]["sampling"]
            two = b["2opt"]
            gap_g = (g - two) / two * 100
            gap_s = (s - two) / two * 100
            print(f"{name:<11}{b['nn']:>9.4f}{two:>9.4f}{g:>9.4f}{s:>9.4f}"
                  f"{gap_g:>+9.2f}%{gap_s:>+9.2f}%")
        print("-" * 78)
        print("Gap columns = (neural - 2Opt)/2Opt. Positive => neural WORSE than 2-Opt.")
    else:
        hdr = f"{'dist':<11}{'NN':>9}{'2-Opt':>9}{'(NN-2Opt)/2Opt':>16}"
        print(hdr)
        print("-" * 50)
        for name in DISTRIBUTIONS:
            b = baseline[name]
            gap = (b["nn"] - b["2opt"]) / b["2opt"] * 100
            print(f"{name:<11}{b['nn']:>9.4f}{b['2opt']:>9.4f}{gap:>+15.2f}%")
        print("-" * 50)
        print("(Neural columns skipped: provide actor.pt to include them.)")
    note = "uniform = in-distribution control; clustered/grid/ring = out-of-distribution."
    print(note)
    if neural_src == "model.pt":
        print("NOTE: neural row used model.pt (supervised stand-in), NOT the RL actor.pt.")


if __name__ == "__main__":
    main()
