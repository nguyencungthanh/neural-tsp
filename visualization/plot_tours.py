import sys
import torch
from visualize import plot_three_tours
sys.path.append("../python")
from model import PointerNet
from dataset import TSPDataset
from heuristics import nearest_neighbor, two_opt
import random
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load one test instance
dataset = TSPDataset("../data/tsp_n12_supervised.txt")
points, optimal_tour = dataset[randint(1, 10000)]

# Load trained model
model = PointerNet().to(device)
model.load_state_dict(torch.load("../python/model.pt", map_location=device))
model.eval()

with torch.no_grad():
    pts_batch = points.unsqueeze(0).to(device)
    _, pred_tour, _ = model(pts_batch)

pred_tour = pred_tour.squeeze(0).cpu()

# Classical heuristics
nn_tour = nearest_neighbor(points)
opt2_tour = two_opt(points, nn_tour)

# Plot comparison
plot_three_tours(
    points,
    optimal_tour,
    pred_tour,
    opt2_tour,
    "Optimal (Held-Karp)",
    "Model Prediction",
    "2-Opt Heuristic"
)

