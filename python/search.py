import torch
from model import PointerNet
from eval_utils import tour_length


def sample_tours(model, points, num_samples=1280, temperature=2.0,
                 batch_size=256,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    RL pretraining-Sampling: sample multiple tours from a fixed policy
    and return the shortest one found.

    No parameter updates — fully parallelizable.

    Args:
        model:        trained PointerNet
        points:       (n, 2) city coordinates — a single test instance
        num_samples:  total number of candidate tours to sample
        temperature:  softmax temperature (>1 = more diverse, paper uses T=1.5-2.2)
        batch_size:   samples per forward pass (for memory efficiency)
        device:       "cuda" or "cpu"

    Returns:
        best_tour:   (n,) best tour found
        best_length: float, length of the best tour
    """
    model.eval()
    model.to(device)
    n = points.size(0)
    points = points.to(device)

    best_tour = None
    best_length = float("inf")

    remaining = num_samples

    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)

            # Replicate the single instance into a batch
            points_batch = points.unsqueeze(0).expand(bs, -1, -1)  # (bs, n, 2)

            # Sample tours with temperature (sample=True so we sample even in eval mode)
            _, tours, _ = model(points_batch, temperature=temperature, sample=True)

            # Compute tour lengths
            lengths = tour_length(points_batch, tours)

            # Track best
            min_idx = lengths.argmin()
            min_length = lengths[min_idx].item()
            if min_length < best_length:
                best_length = min_length
                best_tour = tours[min_idx].cpu()

            remaining -= bs

    return best_tour, best_length


def greedy_decode(model, points,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Greedy decoding: always pick the city with highest probability.
    Fastest inference, no sampling.

    Args:
        model:  trained PointerNet
        points: (n, 2) city coordinates

    Returns:
        tour:   (n,) predicted tour
        length: float, tour length
    """
    model.eval()
    model.to(device)
    points = points.to(device)

    with torch.no_grad():
        points_batch = points.unsqueeze(0)  # (1, n, 2)
        _, tours, _ = model(points_batch)
        tour = tours[0].cpu()
        length = tour_length(points_batch, tours).item()

    return tour, length


