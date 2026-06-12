import torch
import torch.nn as nn
from model import Glimpse


class CriticNet(nn.Module):
    """
    Critic network for Actor-Critic training (Section 4, Algorithm 1).

    Predicts the expected tour length b(s) for a given input graph.
    Architecture: LSTM encoder -> Process block (P glimpse steps) -> 2-layer ReLU decoder.

    Loss: MSE between predicted baseline and actual sampled tour lengths (Eq. 6).
    """

    def __init__(self, input_dim=2, embed_dim=128, hidden_dim=128, process_steps=3):
        super().__init__()

        # Same encoder architecture as PointerNet
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Process block: P steps of glimpse over encoder memory
        self.process_steps = process_steps
        self.glimpse = Glimpse(hidden_dim)

        # Decoder: 2 fully-connected layers (hidden_dim -> hidden_dim -> 1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: (B, n, 2) city coordinates
        Returns: (B,) predicted expected tour length for each graph
        """
        embedded = self.embedding(x)                        # (B, n, d)
        enc_out, (h, c) = self.encoder(embedded)            # enc_out: (B, n, d)

        # Process block: start from final hidden state, glimpse P times
        g = h.squeeze(0)                                     # (B, d)
        for _ in range(self.process_steps):
            g = self.glimpse(enc_out, g)

        # Decode to scalar baseline
        baseline = self.decoder(g).squeeze(-1)               # (B,)
        return baseline
