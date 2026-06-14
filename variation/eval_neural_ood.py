"""
Neural evaluation on the OOD datasets.

Loads a PointerNet policy and runs greedy + sampling decoding on each
distribution file, reporting average tour length.

Model resolution (first found wins):
  1. --model PATH                (explicit)
  2. actor.pt                    (the project's RL policy -> the definitive model)
  3. model.pt                    (supervised pretrain; used as a stand-in if actor.pt
                                  is absent, e.g. on a machine without the trained policy)

NOTE: the project's headline results use actor.pt (trained on uniform TSP-20).
Locally only model.pt (supervised, n=12) may be present, so this script degrades
gracefully and clearly labels which weights were used.

Reuses the parent project's model/search code (no duplication):
  python/model.py, python/search.py, python/dataset_rl.py

Run:
  python variation/eval_neural_ood.py                       # all distributions, defaults
  python variation/eval_neural_ood.py --distributions uniform clustered --samples 1280
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))  # reuse parent project code

import torch
from model import PointerNet
from search import greedy_decode, sample_tours
from dataset_rl import TSPDatasetRL

DATA_DIR = os.path.join(HERE, "data")
PYTHON_DIR = os.path.join(HERE, "..", "python")


def load_policy(model_path=None, device="cpu"):
    """Load a PointerNet. Returns (model, source_name) or (None, None).

    Tries candidates in priority order; skips any that are missing or whose
    architecture doesn't match the current PointerNet (e.g. a stale model.pt
    from an older version of model.py).
    """
    candidates = []
    if model_path:
        candidates.append(model_path)
    candidates += [os.path.join(PYTHON_DIR, "actor.pt"),
                   "actor.pt",
                   os.path.join(PYTHON_DIR, "model.pt"),
                   "model.pt"]
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if not os.path.exists(path):
            continue
        model = PointerNet(embed_dim=128, hidden_dim=128).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            return model, os.path.basename(path)
        except RuntimeError as e:
            # Incompatible architecture (stale checkpoint) -- skip, keep looking.
            print(f"  [skip] {os.path.basename(path)}: architecture mismatch ({e.__class__.__name__})")
    return None, None


def evaluate(data_path, model, device="cpu", num_instances=None,
             num_samples=1280, temperature=1.5, verbose=True):
    """Run greedy + sampling on one distribution file. Returns dict of averages."""
    dataset = TSPDatasetRL(data_path)
    n_inst = len(dataset) if num_instances is None else min(num_instances, len(dataset))

    t0 = time.time()
    total_greedy = 0.0
    total_sample = 0.0
    for i in range(n_inst):
        points = dataset[i]
        _, len_g = greedy_decode(model, points, device=device)
        _, len_s = sample_tours(model, points, num_samples=num_samples,
                                temperature=temperature, device=device)
        total_greedy += len_g
        total_sample += len_s
        if verbose and (i + 1) % max(1, n_inst // 5) == 0:
            print(f"    [{os.path.basename(data_path)}] {i+1}/{n_inst} "
                  f"({time.time()-t0:.1f}s)")
    return {
        "greedy": total_greedy / n_inst,
        "sampling": total_sample / n_inst,
        "n": n_inst,
    }


def main():
    ap = argparse.ArgumentParser(description="Neural (greedy+sampling) eval on OOD datasets.")
    ap.add_argument("--distributions", nargs="+",
                    default=["uniform", "clustered", "grid", "ring"])
    ap.add_argument("--samples", type=int, default=1280, help="tours sampled per instance")
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--instances", type=int, default=None,
                    help="cap to first N instances (use a small N for a quick CPU smoke test)")
    ap.add_argument("--model", default=None, help="explicit weights path")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, src = load_policy(args.model, device=device)
    if model is None:
        print("No policy weights found (actor.pt / model.pt). Skipping neural eval.")
        print("On Colab, ensure python/actor.pt exists and re-run.")
        return

    print(f"Loaded policy: {src}  (device={device})")
    if src == "model.pt":
        print("  WARNING: using supervised model.pt (n=12) as a stand-in; the project's "
              "headline policy is actor.pt. Numbers are a pipeline smoke-test, not the study result.")

    model.eval()
    print(f"\n{'dist':<12}{'greedy':>10}{'sampling':>12}{'N':>8}")
    print("-" * 44)
    for name in args.distributions:
        path = os.path.join(DATA_DIR, f"{name}.txt")
        if not os.path.exists(path):
            print(f"  {name}: missing {path} -- run generate_ood.py first")
            continue
        res = evaluate(path, model, device=device, num_instances=args.instances,
                       num_samples=args.samples, temperature=args.temperature)
        print(f"{name:<12}{res['greedy']:>10.4f}{res['sampling']:>12.4f}{res['n']:>8}")


if __name__ == "__main__":
    main()
