import torch
from torch.utils.data import DataLoader
from dataset import TSPDataset
from model import PointerNet
from eval_utils import tour_length

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TSPDataset("../data/tsp_n12_supervised.txt")
loader = DataLoader(dataset, batch_size=128)

model = PointerNet().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

total_gap = 0
count = 0

with torch.no_grad():
    for points, optimal_tour in loader:
        points = points.to(device)
        optimal_tour = optimal_tour.to(device)

        _, pred_tour, _ = model(points)  # không teacher forcing

        pred_len = tour_length(points, pred_tour)
        opt_len = tour_length(points, optimal_tour)

        gap = (pred_len - opt_len) / opt_len
        total_gap += gap.sum().item()
        count += points.size(0)

print(f"Average optimality gap: {100 * total_gap / count:.2f}%")
