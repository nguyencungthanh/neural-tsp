import torch
from torch.utils.data import DataLoader
from dataset import TSPDataset
from model import PointerNet
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TSPDataset("../data/train_supervised.txt")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = PointerNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    total_loss = 0
    for points, tours in loader:
        points, tours = points.to(device), tours.to(device)

        probs, _, _ = model(points, tours)            # post-softmax distributions
        # Use NLL with log-probs: the model returns probabilities, not raw logits,
        # so F.cross_entropy (which re-applies log_softmax) would double-softmax.
        log_probs = torch.log(probs.clamp_min(1e-30))
        loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), tours.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pt") 