import torch
from torch.utils.data import DataLoader
from dataset_rl import TSPDatasetRL
from model import PointerNet
from eval_utils import tour_length

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TSPDatasetRL("../data/eval.txt")
loader = DataLoader(dataset, batch_size=256)

model = PointerNet(embed_dim=128, hidden_dim=128).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

total_len = 0
count = 0

with torch.no_grad():
    for points in loader:
        points = points.to(device)
        _, tours, _ = model(points)
        lengths = tour_length(points, tours)
        total_len += lengths.sum().item()
        count += points.size(0)

print(f"Model average tour length: {total_len / count:.4f}")
