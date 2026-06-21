"""
Out-of-distribution (OOD) TSP data generator for the distribution-shift study.

Produces four n=20 Euclidean TSP datasets in the SAME text format the rest of
the project uses ("num_instances n" header, then one "x y" line per city):

  uniform    - i.i.d. U[0,1]^2          (the TRAINING distribution -> in-distribution control)
  clustered  - 5 Gaussian blobs          (structure the model never saw)
  grid       - jittered lattice          (regular structure)
  ring       - points on a noisy circle  (1-D manifold)

All coordinates are clipped to [0, 1]. Fixed seeds -> reproducible.
Each instance's city order is independently shuffled so no distribution is
trivially pre-sorted (the policy is order-sensitive, and ring/grid sorted would
otherwise hand the model the optimal tour).

Run:
  python variation/generate_ood.py                 # all 4 distributions, default 1000 x 20
  python variation/generate_ood.py --num 500 --n 20
"""
import os
import argparse
import numpy as np


# --------------------------------------------------------------------------
# Distributions. Each returns an array of shape (num, n, 2) in "natural" order;
# the caller shuffles each instance afterwards.
# --------------------------------------------------------------------------
def gen_uniform(num, n, rng):
    return rng.uniform(0.0, 1.0, size=(num, n, 2))


def gen_clustered(num, n, rng, k=5, sigma=0.07):
    """k well-separated Gaussian blobs; cities assigned to blobs uniformly."""
    out = np.empty((num, n, 2))
    for i in range(num):
        centers = rng.uniform(0.15, 0.85, size=(k, 2))
        assign = rng.integers(0, k, size=n)
        pts = centers[assign] + rng.normal(0.0, sigma, size=(n, 2))
        out[i] = np.clip(pts, 0.0, 1.0)
    return out


def gen_grid(num, n, rng, sigma=0.02):
    """Jittered near-square lattice covering [0.08, 0.92]^2."""
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    xs = np.linspace(0.08, 0.92, cols)
    ys = np.linspace(0.08, 0.92, rows)
    base = np.array([[x, y] for y in ys for x in xs])[:n]  # (n, 2)
    out = np.empty((num, n, 2))
    for i in range(num):
        out[i] = np.clip(base + rng.normal(0.0, sigma, size=(n, 2)), 0.0, 1.0)
    return out


def gen_ring(num, n, rng, sigma=0.015):
    """Points on a circle of radius 0.4 centered at (0.5, 0.5), with angle noise."""
    R = 0.4
    theta0 = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    out = np.empty((num, n, 2))
    for i in range(num):
        theta = theta0 + rng.normal(0.0, sigma, size=n)
        out[i, :, 0] = 0.5 + R * np.cos(theta)
        out[i, :, 1] = 0.5 + R * np.sin(theta)
    return np.clip(out, 0.0, 1.0)


def gen_two_moons(num, n, rng, noise=0.03):
    """Two interleaving half-circles (sklearn-style moons) - a non-convex
    distribution none of uniform/clustered/grid/ring reproduce. Each city is
    assigned to the upper or lower moon at random; the angle is uniform on [0, pi].
    Output is rescaled into [0, 1]^2 (independent of n and noise).
    """
    out = np.empty((num, n, 2))
    for i in range(num):
        upper = rng.random(n) < 0.5                       # which moon each city belongs to
        theta = rng.uniform(0.0, np.pi, size=n)           # angle along the half-circle
        x = np.where(upper, np.cos(theta), 1.0 - np.cos(theta))
        y = np.where(upper, np.sin(theta), 0.5 - np.sin(theta))
        pts = np.stack([x, y], axis=1) + rng.normal(0.0, noise, size=(n, 2))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        out[i] = np.clip((pts - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return out


GENERATORS = {
    "uniform": gen_uniform,
    "clustered": gen_clustered,
    "grid": gen_grid,
    "ring": gen_ring,
    "two_moons": gen_two_moons,
}


def _shuffle_order(pts, rng):
    """Independently shuffle the city order within each instance."""
    num, n, _ = pts.shape
    for i in range(num):
        pts[i] = pts[i][rng.permutation(n)]
    return pts


# Stable per-distribution seed offsets. (Do NOT use hash(name) -- Python's
# string hash is randomized per process via PYTHONHASHSEED, which would make
# generation non-reproducible across runs.)
SEED_OFFSETS = {"uniform": 0, "clustered": 1, "grid": 2, "ring": 3, "two_moons": 4}


def generate_distribution(name, num, n, seed=0):
    """Return (num, n, 2) float array for the named distribution."""
    if name not in GENERATORS:
        raise ValueError(f"unknown distribution '{name}'; choose from {list(GENERATORS)}")
    off = SEED_OFFSETS[name]
    # Independent per-distribution seeds so adding/removing one doesn't reseed the others.
    rng = np.random.default_rng(seed + off)
    pts = GENERATORS[name](num, n, rng)
    rng2 = np.random.default_rng(seed + 100 + off)
    return _shuffle_order(pts, rng2)


def write_tsp_file(path, pts):
    """Write (num, n, 2) array in project text format."""
    num, n, _ = pts.shape
    with open(path, "w") as f:
        f.write(f"{num} {n}\n")
        for inst in pts:
            for x, y in inst:
                f.write(f"{x:.6f} {y:.6f}\n")


def read_tsp_file(path):
    """Read back a project-format file -> (num, n, 2) array (for validation)."""
    with open(path) as f:
        first = f.readline().split()
        num, n = int(first[0]), int(first[1])
        pts = np.empty((num, n, 2))
        for i in range(num):
            for j in range(n):
                pts[i, j] = [float(v) for v in f.readline().split()]
    return pts


def mean_nn_distance(pts):
    """Mean nearest-neighbour distance per instance, averaged over instances.
    A cheap 'is this distribution actually different' diagnostic:
    clustered/grid have small NN distance; uniform is larger."""
    num, n, _ = pts.shape
    tot = 0.0
    for i in range(num):
        p = pts[i]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)  # (n, n)
        np.fill_diagonal(d, np.inf)
        tot += d.min(axis=1).mean()
    return tot / num


def validate(path, expected_num, expected_n):
    """Read back and sanity-check a generated file. Raises on any problem."""
    pts = read_tsp_file(path)
    num, n, _ = pts.shape
    assert num == expected_num, f"{path}: num {num} != {expected_num}"
    assert n == expected_n, f"{path}: n {n} != {expected_n}"
    assert np.isfinite(pts).all(), f"{path}: non-finite coordinate present"
    assert pts.min() >= 0.0 and pts.max() <= 1.0, f"{path}: coords out of [0,1]"
    return pts


def main():
    ap = argparse.ArgumentParser(description="Generate OOD TSP datasets for the distribution-shift study.")
    ap.add_argument("--num", type=int, default=1000, help="instances per distribution")
    ap.add_argument("--n", type=int, default=20, help="cities per instance")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                    help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distributions", nargs="+", default=list(GENERATORS),
                    help="subset to generate")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Generating {args.num} x {args.n} instances each, seed={args.seed}")
    print(f"{'dist':<12}{'mean NN dist':>14}  file")
    print("-" * 60)
    for name in args.distributions:
        pts = generate_distribution(name, args.num, args.n, seed=args.seed)
        path = os.path.join(args.out, f"{name}.txt")
        write_tsp_file(path, pts)
        # validate by reading straight back from disk
        reread = validate(path, args.num, args.n)
        nn = mean_nn_distance(reread)
        print(f"{name:<12}{nn:>14.4f}  {os.path.relpath(path)}")
    print("\nAll files validated: correct shape, finite, coords in [0,1].")


if __name__ == "__main__":
    main()
