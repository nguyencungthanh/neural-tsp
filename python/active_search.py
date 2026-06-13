import torch
from model import PointerNet
from eval_utils import tour_length


def active_search(points, model=None, state_dict_path=None,
                  num_steps=10000, batch_size=128, lr=1e-5,
                  alpha=0.99, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Active Search (Algorithm 2 from the paper).

    Optimizes the policy parameters on a single test instance,
    tracking the best solution found during search.

    Args:
        points:          (n, 2) city coordinates — a single test instance
        model:           PointerNet instance (optional, used if provided)
        state_dict_path: path to pretrained weights (optional, e.g. "actor.pt")
        num_steps:       number of search iterations
        batch_size:      number of candidate tours sampled per step
        lr:              learning rate for Adam
        alpha:           baseline decay for exponential moving average
        device:          "cuda" or "cpu"

    Returns:
        best_tour:  (n,) best tour found
        best_length: float, length of the best tour
    """
    n = points.size(0)
    points = points.to(device)

    # Initialize model
    if model is None:
        model = PointerNet(embed_dim=128, hidden_dim=128).to(device)

    # Load pretrained weights if provided
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path, map_location=device))

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Replicate the single instance into a batch
    points_batch = points.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n, 2)

    best_tour = None
    best_length = float("inf")
    baseline = None

    for step in range(num_steps):
        # Input shuffling: randomly permute city order per sample in batch
        perms = torch.stack([torch.randperm(n) for _ in range(batch_size)]).to(device)  # (B, n)
        idx = perms.unsqueeze(-1).expand(-1, -1, 2)  # (B, n, 2)
        points_shuffled = torch.gather(points_batch, 1, idx)

        # Sample tours from policy
        _, tours, log_probs = model(points_shuffled)

        # Compute tour lengths on the shuffled points
        lengths = tour_length(points_shuffled, tours)

        # Track best solution found so far. Map the tour back from the shuffled
        # space to original city indices so the returned tour is directly usable.
        min_idx = lengths.argmin()
        min_length = lengths[min_idx].item()
        if min_length < best_length:
            best_length = min_length
            best_tour = perms[min_idx][tours[min_idx]].cpu()  # original-space tour

        # Exponential moving average baseline (Eq. in Algorithm 2)
        if baseline is None:
            baseline = lengths.mean().item()
        else:
            baseline = alpha * baseline + (1 - alpha) * lengths.mean().item()

        # Policy gradient with moving average baseline
        advantage = lengths - baseline  # positive = worse than average
        loss = (advantage.detach() * log_probs.sum(dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1}/{num_steps}, "
                  f"Best length: {best_length:.4f}, "
                  f"Baseline: {baseline:.4f}")

    return best_tour, best_length


if __name__ == "__main__":
    from dataset_rl import TSPDatasetRL

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a test instance
    dataset = TSPDatasetRL("../data/eval.txt")
    points = dataset[0]  # (n, 2)

    print("=== Active Search (from scratch) ===")
    tour_as, len_as = active_search(
        points, num_steps=10000, batch_size=128, lr=1e-5, device=device
    )
    print(f"Best tour length: {len_as:.4f}\n")

    print("=== RL pretraining + Active Search ===")
    tour_rl_as, len_rl_as = active_search(
        points, state_dict_path="actor.pt",
        num_steps=10000, batch_size=128, lr=1e-5, device=device
    )
    print(f"Best tour length: {len_rl_as:.4f}")
