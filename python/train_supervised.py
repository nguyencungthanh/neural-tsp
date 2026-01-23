import torch
from torch.utils.data import DataLoader
from dataset import TSPDataset
from model import PointerNet
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TSPDataset("../data/tsp_n12_supervised.txt")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = PointerNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    total_loss = 0
    for points, tours in loader:
        points, tours = points.to(device), tours.to(device)

        logits, _, _ = model(points, tours)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tours.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pt") 