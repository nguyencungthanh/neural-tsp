import torch
import torch.nn as nn
import torch.nn.functional as F

class Glimpse(nn.Module):
    """
    Glimse mechanism (Appendix A.1, Eq. 11-14).
    Aggregates reference vectors via attention before the final pointing step.
    G(ref, q) = sum_i r_i * softmax(v_g^T tanh(W_ref_g r_i + W_q_g q))
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_ref_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_g = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, ref, q):
        """
        ref: (B, n, d) — encoder outputs
        q:   (B, d)    — query vector
        Returns: (B, d) — glimpsed query
        """
        # (B, n, d) + (B, 1, d) -> (B, n, d)
        scores = self.v_g(torch.tanh(
            self.W_ref_g(ref) + self.W_q_g(q).unsqueeze(1)
        )).squeeze(-1)  # (B, n)
        probs = torch.softmax(scores, dim=1)  # (B, n)
        return torch.bmm(probs.unsqueeze(1), ref).squeeze(1)  # (B, d)


class PointerNet(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, hidden_dim=128,
                 glimpse_steps=1, clip_logits=10.0):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Parameterized pointing attention (Eq. 8-9):
        #   u_i = v^T tanh(W_ref r_i + W_q q)
        self.W_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # Glimpse mechanism (Eq. 11-14)
        self.glimpse_steps = glimpse_steps
        self.glimpse = Glimpse(hidden_dim)

        # Logit clipping (Eq. 16): softmax(C * tanh(u))
        self.clip_logits = clip_logits

    def _pointing(self, ref, q, mask, temperature=1.0):
        """
        Compute pointing distribution over references.
        ref:  (B, n, d) — encoder outputs
        q:    (B, d)    — current decoder query (after glimpse)
        mask: (B, n)    — True for cities already visited
        temperature: softmax temperature (Appendix A.2, Eq. 15)
        Returns: (B, n) probabilities
        """
        # Eq. 8: u_i = v^T tanh(W_ref r_i + W_q q)
        scores = self.v(torch.tanh(
            self.W_ref(ref) + self.W_q(q).unsqueeze(1)
        )).squeeze(-1)  # (B, n)

        # Eq. 16: logit clipping (applied to unmasked logits)
        if self.clip_logits > 0:
            scores = self.clip_logits * torch.tanh(scores)

        # Mask already-visited cities AFTER clipping, using -inf so softmax gives
        # them exactly zero probability. (Masking before the tanh clip would leave
        # residual mass: C*tanh(-1e9) = -C, not -inf, so a masked city could be sampled.)
        scores = scores.masked_fill(mask, float("-inf"))

        # Eq. 15: temperature
        probs = torch.softmax(scores / temperature, dim=1)
        return probs

    def forward(self, x, target=None, temperature=1.0, sample=False):
        """
        x:           (B, n, 2) city coordinates
        target:      (B, n) tour indices for teacher forcing (supervised training)
        temperature: softmax temperature (1.0 during training, >1 for exploration)
        sample:      if True, sample from the policy even in eval mode (Sampling
                     inference). Ignored when target is given; training
                     (self.training) always samples regardless of this flag.
        """
        B, n, _ = x.size()

        embedded = self.embedding(x)  # (B, n, d)
        enc_out, (h, c) = self.encoder(embedded)  # enc_out: (B, n, d)

        # Trainable start token for first decoder step
        dec_input = torch.zeros(B, 1, enc_out.size(-1), device=x.device)
        mask = torch.zeros(B, n, device=x.device).bool()

        outputs = []
        tours = []
        log_probs = []

        for t in range(n):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            query = dec_out.squeeze(1)  # (B, d)

            # Glimpse: apply glimpse steps before pointing (Eq. 13-14)
            g = query
            for _ in range(self.glimpse_steps):
                g = self.glimpse(enc_out, g)

            # Pointing distribution (Eq. 8-10)
            probs = self._pointing(enc_out, g, mask, temperature)

            # Select next city
            if target is not None:
                # Teacher forcing (supervised training)
                next_idx = target[:, t]
                log_prob = None
            elif self.training or sample:
                # Sample from policy (RL training, or Sampling inference)
                dist = torch.distributions.Categorical(probs)
                next_idx = dist.sample()
                log_prob = dist.log_prob(next_idx)
            else:
                # Greedy decode (inference)
                next_idx = probs.argmax(dim=1)
                log_prob = None

            if log_prob is not None:
                log_probs.append(log_prob)

            tours.append(next_idx)

            mask = mask.clone()
            mask[torch.arange(B, device=x.device), next_idx] = True
            dec_input = enc_out[torch.arange(B, device=x.device), next_idx].unsqueeze(1)

            outputs.append(probs)

        outputs = torch.stack(outputs, dim=1)  # (B, n, n)
        tours = torch.stack(tours, dim=1)      # (B, n)
        log_probs = torch.stack(log_probs, dim=1) if log_probs else None
        return outputs, tours, log_probs

    @staticmethod
    def tsp_loss(probs, target):
        """Cross-entropy loss for supervised training.

        NOTE: the model returns post-softmax probabilities, so we take log and use
        NLLLoss. F.cross_entropy would re-apply log_softmax (double-softmax).
        """
        B, n, _ = probs.shape
        log_probs = torch.log(probs.clamp_min(1e-30))
        return F.nll_loss(log_probs.view(B * n, -1), target.view(-1))
