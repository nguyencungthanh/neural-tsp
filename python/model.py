import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.pointer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, target=None):
        """
        x: (B, n, 2)
        target: (B, n) tour indices (for teacher forcing)
        """
        B, n, _ = x.size()

        embedded = self.embedding(x)
        enc_out, (h, c) = self.encoder(embedded)

        dec_input = torch.zeros(B, 1, enc_out.size(-1), device=x.device)
        mask = torch.zeros(B, n, device=x.device).bool()

        outputs = []
        tours = []
        log_probs = []

        for t in range(n):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            query = self.pointer(dec_out.squeeze(1))  # (B,H)

            scores = torch.bmm(enc_out, query.unsqueeze(2)).squeeze(2)  # (B,n)
            scores = scores.masked_fill(mask, -1e9)
            
            probs = torch.softmax(scores, dim=1)

            # Use teacher forcing during training if target is provided
            if target is not None:
                next_idx = target[:, t]
            else:
                if self.training:
                    dist = torch.distributions.Categorical(probs)
                    next_idx = dist.sample()
                    log_prob = dist.log_prob(next_idx)
                else: 
                    next_idx = probs.argmax(dim=1)
                    log_prob = None

                if log_prob is not None:
                    log_probs.append(log_prob)

            tours.append(next_idx)
        
            mask = mask.clone()
            mask[torch.arange(B), next_idx] = True
            dec_input = enc_out[torch.arange(B), next_idx].unsqueeze(1)

            outputs.append(scores) 

        outputs = torch.stack(outputs, dim=1)
        tours = torch.stack(tours, dim=1)
        log_probs = torch.stack(log_probs, dim=1) if log_probs else None
        return outputs, tours, log_probs  # (B, n, n)

    def tsp_loss(logits, target):
        B, n, _ = logits.shape
        return F.cross_entropy(
            logits.view(B*n, -1),
            target.view(-1)
        )

