import torch
from torch.utils.data import DataLoader
from dataset_rl import TSPDatasetRL
from model import PointerNet
from eval_utils import tour_length
from search import greedy_decode, sample_tours
from active_search import active_search

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test data
dataset = TSPDatasetRL("../data/eval.txt")
n_instances = len(dataset)

# Load trained actor
actor = PointerNet(embed_dim=128, hidden_dim=128).to(device)
try:
    actor.load_state_dict(torch.load("actor.pt", map_location=device))
    print("Loaded actor.pt")
except FileNotFoundError:
    print("WARNING: actor.pt not found. Skipping pretrained methods.")

# ============================================================
# 1. RL pretraining-Greedy
# ============================================================
print("\n=== RL pretraining-Greedy ===")
total_len = 0
for i in range(n_instances):
    points = dataset[i]
    _, length = greedy_decode(actor, points, device=device)
    total_len += length
print(f"Average tour length: {total_len / n_instances:.4f}")

# ============================================================
# 2. RL pretraining-Sampling (with tuned temperature)
# ============================================================
print("\n=== RL pretraining-Sampling ===")
for num_samples in [128, 1280, 12800]:
    for temp in [1.5, 2.0, 2.2]:
        total_len = 0
        for i in range(n_instances):
            points = dataset[i]
            _, length = sample_tours(
                actor, points, num_samples=num_samples,
                temperature=temp, device=device
            )
            total_len += length
        avg = total_len / n_instances
        print(f"  T={temp}, N={num_samples:>6}: avg length = {avg:.4f}")

# ============================================================
# 3. RL pretraining-Active Search (on a few instances)
# ============================================================
print("\n=== RL pretraining-Active Search (first 5 instances) ===")
n_active = min(5, n_instances)
total_len = 0
for i in range(n_active):
    points = dataset[i]
    tour, length = active_search(
        points, state_dict_path="actor.pt",
        num_steps=5000, batch_size=128, lr=1e-5, device=device
    )
    total_len += length
    print(f"  Instance {i+1}: length = {length:.4f}")
print(f"  Average (first {n_active}): {total_len / n_active:.4f}")

# ============================================================
# 4. Active Search from scratch (on a few instances)
# ============================================================
print("\n=== Active Search from scratch (first 3 instances) ===")
n_scratch = min(3, n_instances)
total_len = 0
for i in range(n_scratch):
    points = dataset[i]
    tour, length = active_search(
        points, num_steps=5000, batch_size=128, lr=1e-4, device=device
    )
    total_len += length
    print(f"  Instance {i+1}: length = {length:.4f}")
print(f"  Average (first {n_scratch}): {total_len / n_scratch:.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print("Run baselines for comparison:")
print("  g++ -std=c++17 ../cpp/baselines.cpp -o baselines")
print("  ./baselines < ../data/eval.txt")
print("=" * 50)
