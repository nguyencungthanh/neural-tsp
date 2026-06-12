import torch
from torch.utils.data import DataLoader
from dataset_rl import TSPDatasetRL
from model import PointerNet
from eval_utils import tour_length

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TSPDatasetRL("../data/train_rl.txt")
loader = DataLoader(dataset, batch_size=256, shuffle=True)

model = PointerNet(embed_dim=128, hidden_dim=128).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))  # pretrained
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
baseline = None
beta = 0.9  # moving average baseline

for epoch in range(5):
    total_reward = 0

    for points in loader:
        points = points.to(device)

        _, tours, log_probs = model(points)


        lengths = tour_length(points, tours)
        reward = -lengths

        if baseline is None:
            baseline = reward.mean()
        else:
            baseline = beta * baseline + (1 - beta) * reward.mean()

        advantage = reward - baseline

        loss = -(advantage.detach() * log_probs.sum(dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward.mean().item()

    print(f"Epoch {epoch+1}, Avg tour length: {-total_reward/len(loader):.4f}")
