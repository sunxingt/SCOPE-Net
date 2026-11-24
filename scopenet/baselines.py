"""Same-backbone open-set baselines: RPL (reciprocal points) and PROSER (dummy classifier + manifold mixup).

Both use the Plain3D encoder of SCOPE-Net and a threshold from the known validation quantile.
Simplified re-implementations for a controlled comparison; they are not the authors' code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Plain3D


class RPLNet(nn.Module):
    """Reciprocal point learning: logit_k = ||f - p_k||^2 / D (distance to the reciprocal point of class k)."""

    def __init__(self, bands, patch, C, dim=128, margin_w=0.1):
        super().__init__()
        self.enc = Plain3D(bands, patch, dim)
        self.points = nn.Parameter(0.1 * torch.randn(C, dim))
        self.radius = nn.Parameter(torch.zeros(1))
        self.margin_w, self.dim = margin_w, dim

    def logits(self, x, raw):
        s, c, _, _ = self.enc(x, raw)
        f = s + c
        d = ((f[:, None] - self.points[None]) ** 2).mean(-1)   # N,C
        return d, f

    def loss(self, x, raw, y):
        d, f = self.logits(x, raw)
        ce = F.cross_entropy(d, y)
        dist_true = d[torch.arange(len(y)), y]
        reg = F.mse_loss(dist_true, self.radius.expand_as(dist_true))
        return ce + self.margin_w * reg

    def unknown_score(self, x, raw):
        d, _ = self.logits(x, raw)
        return -d.max(1).values, d          # far from every reciprocal point -> known


class PROSERNet(nn.Module):
    """Dummy-classifier placeholder learning with manifold mixup data placeholders."""

    def __init__(self, bands, patch, C, dim=128, beta=1.0, gamma=1.0):
        super().__init__()
        self.enc = Plain3D(bands, patch, dim)
        self.head = nn.Linear(dim, C + 1)
        self.C, self.beta, self.gamma = C, beta, gamma

    def feat(self, x, raw):
        s, c, _, _ = self.enc(x, raw)
        return s + c

    def loss(self, x, raw, y):
        f = self.feat(x, raw)
        lg = self.head(f)
        ce = F.cross_entropy(lg, y)
        # classifier placeholder: with the true class masked, the dummy class should win
        masked = lg.clone()
        masked[torch.arange(len(y)), y] = -1e9
        dummy = torch.full_like(y, self.C)
        l_cp = F.cross_entropy(masked, dummy)
        # data placeholder: mixup of embeddings from different classes -> dummy class
        perm = torch.randperm(len(y), device=y.device)
        keep = y != y[perm]
        l_dp = lg.new_zeros(())
        if keep.any():
            lam = torch.distributions.Beta(2.0, 2.0).sample((int(keep.sum()), 1)).to(f.device)
            fm = lam * f[keep] + (1 - lam) * f[perm][keep]
            l_dp = F.cross_entropy(self.head(fm), torch.full((int(keep.sum()),), self.C, device=y.device))
        return ce + self.beta * l_cp + self.gamma * l_dp

    def unknown_score(self, x, raw):
        lg = self.head(self.feat(x, raw))
        p = lg.softmax(1)
        return p[:, self.C], lg[:, :self.C]
