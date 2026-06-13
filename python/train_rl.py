import os
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
CHECKPOINT_EVERY = 5  # save checkpoint every N epochs

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

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

# Learning rate schedulers (paper: decay by 0.96 every 5000 steps)
scheduler_actor = torch.optim.lr_scheduler.StepLR(
    optimizer_actor, step_size=LR_DECAY_STEPS, gamma=LR_DECAY
)
scheduler_critic = torch.optim.lr_scheduler.StepLR(
    optimizer_critic, step_size=LR_DECAY_STEPS, gamma=LR_DECAY
)

start_epoch = 0
global_step = 0

# Resume from latest checkpoint if available
# Resume from latest checkpoint if available. Sort by epoch NUMBER, not lexically,
# otherwise ckpt_epoch20.pt sorts before ckpt_epoch5.pt and resume loads the wrong one.
def _epoch_of(fname):
    return int(fname.replace("ckpt_epoch", "").replace(".pt", ""))

checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
if checkpoints:
    latest = max(checkpoints, key=_epoch_of)
    ckpt = torch.load(f"checkpoints/{latest}", map_location=device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
    optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
    scheduler_actor.load_state_dict(ckpt["scheduler_actor"])
    scheduler_critic.load_state_dict(ckpt["scheduler_critic"])
    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt["global_step"]
    actor.train()
    critic.train()
    print(f"Resumed from checkpoint: {latest} (epoch {start_epoch}, step {global_step})")

for epoch in range(start_epoch, EPOCHS):
    total_reward = 0
    num_batches = 0

    for points in loader:
        points = points.to(device)
        B = points.size(0)
        n = points.size(1)

        # Input shuffling: randomly permute city order PER INSTANCE (paper Sec 4),
        # so each instance in the batch gets a different city ordering.
        perms = torch.stack([torch.randperm(n, device=device) for _ in range(B)])
        idx = perms.unsqueeze(-1).expand(-1, -1, 2)
        points_shuffled = torch.gather(points, 1, idx)

        # Sample tours from actor (Eq. 2-3)
        _, tours, log_probs = actor(points_shuffled)

        # Compute tour lengths L(pi|s) (Eq. 1)
        lengths = tour_length(points_shuffled, tours)

        # Critic predicts expected tour length b(s) (Eq. 6)
        baseline = critic(points_shuffled)

        # Advantage = -(L - b) since reward = -length
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

        # Step learning rate schedulers
        scheduler_actor.step()
        scheduler_critic.step()

        total_reward += (-lengths).mean().item()
        num_batches += 1
        global_step += 1

    avg_len = -total_reward / num_batches
    lr = optimizer_actor.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg tour length: {avg_len:.4f}, LR: {lr:.6f}, Step: {global_step}")

    # Save checkpoint every N epochs
    if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == EPOCHS:
        ckpt_path = f"checkpoints/ckpt_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_actor": optimizer_actor.state_dict(),
            "optimizer_critic": optimizer_critic.state_dict(),
            "scheduler_actor": scheduler_actor.state_dict(),
            "scheduler_critic": scheduler_critic.state_dict(),
            "avg_tour_length": avg_len,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

# Save final models
torch.save(actor.state_dict(), "actor.pt")
torch.save(critic.state_dict(), "critic.pt")
print("Saved actor.pt and critic.pt")
