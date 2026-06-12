import torch
from torch.utils.data import DataLoader
from dataset_rl import TSPDatasetRL
from model import PointerNet
from critic import CriticNet
from eval_utils import tour_length

device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters (matching paper Section 5.1)
BATCH_SIZE = 128
EPOCHS = 20
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
LR_DECAY = 0.96
LR_DECAY_STEPS = 5000
GRAD_CLIP = 1.0

# Load training data
dataset = TSPDatasetRL("../data/train_rl.txt")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize actor (PointerNet) and critic (CriticNet)
actor = PointerNet(embed_dim=128, hidden_dim=128).to(device)
critic = CriticNet(embed_dim=128, hidden_dim=128).to(device)

# Optionally load pretrained supervised model for actor
try:
    actor.load_state_dict(torch.load("model.pt", map_location=device))
    print("Loaded pretrained supervised model for actor.")
except FileNotFoundError:
    print("No pretrained model found — training actor from scratch.")

actor.train()
critic.train()

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

for epoch in range(EPOCHS):
    total_reward = 0
    num_batches = 0

    for points in loader:
        points = points.to(device)
        B = points.size(0)
        n = points.size(1)

        # Input shuffling: randomly permute city order per instance
        perm = torch.randperm(n, device=device)
        points_shuffled = points[:, perm, :]

        # Sample tours from actor (Eq. 2-3)
        _, tours, log_probs = actor(points_shuffled)

        # Compute tour lengths L(pi|s) (Eq. 1)
        lengths = tour_length(points_shuffled, tours)

        # Critic predicts expected tour length b(s) (Eq. 6)
        baseline = critic(points_shuffled)

        # Advantage = -(L - b) since reward = -length
        # We use negative tour length as reward
        advantage = -(lengths - baseline.detach())

        # Actor loss: policy gradient (Eq. 5)
        actor_loss = -(advantage * log_probs.sum(dim=1)).mean()

        # Critic loss: MSE (Eq. 6)
        critic_loss = ((baseline - lengths) ** 2).mean()

        # Update actor
        optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        optimizer_actor.step()

        # Update critic
        optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
        optimizer_critic.step()

        total_reward += (-lengths).mean().item()
        num_batches += 1

    avg_len = -total_reward / num_batches
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg tour length: {avg_len:.4f}")

# Save actor and critic
torch.save(actor.state_dict(), "actor.pt")
torch.save(critic.state_dict(), "critic.pt")
print("Saved actor.pt and critic.pt")
