"""SCOPE-Net modules: CCE encoder (Sec. 3.2), MAPG prototype geometry (Sec. 3.3)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


# ---------------------------------------------------------------- CCE
class SpectralBranch(nn.Module):
    """Eq. (2): centre spectrum + first difference -> 1-D conv -> GAP -> g_s."""

    def __init__(self, dim, use_diff=True):
        super().__init__()
        self.use_diff = use_diff
        cin = 2 if use_diff else 1
        self.net = nn.Sequential(
            nn.Conv1d(cin, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(True))
        self.proj = nn.Sequential(nn.Linear(128, dim), nn.BatchNorm1d(dim))

    def forward(self, xc):  # xc: N,B
        if self.use_diff:
            d = torch.cat([xc[:, 1:] - xc[:, :-1], torch.zeros_like(xc[:, :1])], 1)
            h = torch.stack([xc, d], 1)
        else:
            h = xc[:, None]
        return self.proj(self.net(h).mean(-1))


class PixelFeatures(nn.Module):
    """Two 3-D conv layers -> per-pixel feature F_j in R^D."""

    def __init__(self, bands, dim):
        super().__init__()
        self.c3d = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), stride=(2, 1, 1), padding=(3, 1, 1)), nn.BatchNorm3d(8), nn.ReLU(True),
            nn.Conv3d(8, 16, (5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)), nn.BatchNorm3d(16), nn.ReLU(True))
        with torch.no_grad():
            b = self.c3d(torch.zeros(1, 1, bands, 3, 3)).shape[2]
        self.pw = nn.Sequential(nn.Conv2d(16 * b, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(True))

    def forward(self, x):  # N,B,P,P -> N,P*P,D
        h = self.c3d(x[:, None])
        n, c, b, p, q = h.shape
        h = self.pw(h.reshape(n, c * b, p, q))
        return h.flatten(2).transpose(1, 2)


class CCE(nn.Module):
    """Centre-Consistent Spectral-Spatial Encoder.

    angle_prior: inject log g_j into attention (Eq. 4); if False, g still gives purity.
    angle_on: 'raw' (default, Eq. 3) or 'std' (ablation: angle on standardised data).
    """

    def __init__(self, bands, patch, dim=128, tau_g=0.05, eta0=0.1, angle_prior=True,
                 angle_on="raw", use_spec=True, use_diff=True):
        super().__init__()
        self.P, self.dim, self.tau_g = patch, dim, tau_g
        self.angle_prior, self.angle_on, self.use_spec = angle_prior, angle_on, use_spec
        self.eta = nn.Parameter(torch.tensor(float(eta0)))
        self.spec = SpectralBranch(dim, use_diff) if use_spec else None
        self.pix = PixelFeatures(bands, dim)
        self.Wq, self.Wk, self.Wv = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gc = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        if not use_spec:  # signature space taken from the centre pixel feature instead
            self.gs_alt = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))

    def gate(self, x, raw):
        r = raw if self.angle_on == "raw" else x
        r = r.flatten(2)  # N,B,P*P
        rc = r[:, :, r.shape[-1] // 2:r.shape[-1] // 2 + 1]
        cos = (r * rc).sum(1) / (r.norm(dim=1) * rc.norm(dim=1) + 1e-8)
        theta = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))  # N,P*P
        return torch.sigmoid((self.eta - theta) / self.tau_g)

    def forward(self, x, raw):
        N, B, P, _ = x.shape
        ctr = (P * P) // 2
        g = self.gate(x, raw)                       # N,P*P
        rho = g.mean(1)                             # Eq. (6)
        Fp = self.pix(x)                            # N,P*P,D
        Fc = Fp[:, ctr]
        logits = (self.Wq(Fc)[:, None] * self.Wk(Fp)).sum(-1) / math.sqrt(self.dim)
        if self.angle_prior:
            logits = logits + torch.log(g + 1e-6)  # Eq. (4)
        a = logits.softmax(1)
        c = self.gc(Fc + (a[..., None] * self.Wv(Fp)).sum(1))  # Eq. (5)
        if self.use_spec:
            s = self.spec(x[:, :, P // 2, P // 2])
        else:
            s = self.gs_alt(Fc)
        return s, c, rho, {"gate": g, "attn": a}


class Plain3D(nn.Module):
    """Ablation encoder without CCE: 3-D conv + global pooling, two projection heads, rho == 1."""

    def __init__(self, bands, patch, dim=128):
        super().__init__()
        self.pix = PixelFeatures(bands, dim)
        self.hs = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.hc = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))

    def forward(self, x, raw):
        h = self.pix(x).mean(1)
        return self.hs(h), self.hc(h), torch.ones(x.shape[0], device=x.device), {}


class GatedCCE(nn.Module):
    """Lightweight CCE on the 3-D conv backbone: spectral-angle gated pooling (no learned q/k).

    sig_mode='center': signature space from the centre-pixel feature;
    sig_mode='mean'  : signature space from plain mean pooling (holistic context).
    The context space pools pixel features with weights g_j (Eq. 3), purity rho = mean(g).
    """

    def __init__(self, bands, patch, dim=128, tau_g=0.05, eta0=0.1, sig_mode="mean", gated=True, angle_on="raw"):
        super().__init__()
        self.tau_g, self.sig_mode, self.gated, self.angle_on = tau_g, sig_mode, gated, angle_on
        self.eta = nn.Parameter(torch.tensor(float(eta0)))
        self.pix = PixelFeatures(bands, dim)
        self.hs = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.hc = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))

    def forward(self, x, raw):
        r = (raw if self.angle_on == "raw" else x).flatten(2)
        ctr = r.shape[-1] // 2
        rc = r[:, :, ctr:ctr + 1]
        cos = (r * rc).sum(1) / (r.norm(dim=1) * rc.norm(dim=1) + 1e-8)
        theta = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        g = torch.sigmoid((self.eta - theta) / self.tau_g)
        rho = g.mean(1)
        Fp = self.pix(x)                                   # N,P*P,D
        wgt = g if self.gated else torch.ones_like(g)
        hc = (wgt[..., None] * Fp).sum(1) / wgt.sum(1, keepdim=True).clamp_min(1e-6)
        hs = Fp[:, ctr] if self.sig_mode == "center" else Fp.mean(1)
        return self.hs(hs), self.hc(hc), rho, {"gate": g}


class SCOPENet(nn.Module):
    def __init__(self, bands, patch, n_known, dim=128, encoder="cce", **cce_kw):
        super().__init__()
        if encoder == "cce":
            self.enc = CCE(bands, patch, dim, **cce_kw)
        elif encoder in ("cce2", "cce2c"):
            kw = {k: v for k, v in cce_kw.items() if k in ("tau_g", "eta0", "angle_on")}
            kw["gated"] = cce_kw.get("angle_prior", True)
            self.enc = GatedCCE(bands, patch, dim, sig_mode="center" if encoder == "cce2c" else "mean", **kw)
        else:
            self.enc = Plain3D(bands, patch, dim)
        self.cls = nn.Linear(2 * dim, n_known)

    def forward(self, x, raw):
        s, c, rho, aux = self.enc(x, raw)
        logits = self.cls(torch.cat([s, c], 1))
        return F.normalize(s, dim=1), F.normalize(c, dim=1), rho, logits, aux

    def embed_head(self, s, c):
        return self.cls(torch.cat([s, c], 1))


# ---------------------------------------------------------------- MAPG
def js_div(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps).log() - (b + eps).log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class PrototypeBank(nn.Module):
    """Multi-prototypes with per-mode distance moments (Eqs. 7-11).

    dist_mode: 'mode'   -> per-prototype (mu, sigma) normalisation (default)
               'abs'    -> raw cosine distance (ablation)
               'class'  -> per-class (mu, sigma) shared by its prototypes (ablation)
               'global' -> one global (mu, sigma) (ablation, DSCP-style z-score)
    """

    def __init__(self, n_cls, k, dim, kappa=0.99, sigma_min=0.01, tau_a=0.1, gamma=1.0,
                 omega=0.5, dist_mode="mode", fixed_w=None, loo=True, shrink=0.0):
        super().__init__()
        self.C, self.K, self.kappa, self.smin = n_cls, k, kappa, sigma_min
        self.tau_a, self.gamma, self.omega, self.mode, self.fixed_w = tau_a, gamma, omega, dist_mode, fixed_w
        self.loo = loo
        self.shrink = shrink
        self.register_buffer("P", torch.zeros(2, n_cls, k, dim))
        self.register_buffer("m1", torch.zeros(2, n_cls, k))
        self.register_buffer("m2", torch.zeros(2, n_cls, k))
        self.register_buffer("cnt", torch.ones(n_cls, k))
        self.register_buffer("ready", torch.tensor(False))

    # ---- statistics
    @torch.no_grad()
    def init_kmeans(self, s, c, y, seed=0):
        for cl in range(self.C):
            m = y == cl
            fs, fc = s[m], c[m]
            n = fs.shape[0]
            k = min(self.K, n)
            lab = KMeans(k, n_init=10, random_state=seed).fit_predict(fs.cpu().numpy()) if k > 1 else np.zeros(n, int)
            lab = torch.as_tensor(lab, device=s.device)
            for j in range(self.K):
                sel = lab == (j % k)
                self.P[0, cl, j] = F.normalize(fs[sel].mean(0), dim=0)
                self.P[1, cl, j] = F.normalize(fc[sel].mean(0), dim=0)
        self.recompute_stats(s, c, y)
        self.ready.fill_(True)

    def assign(self, s, y):
        """nearest prototype of the true class in the signature space."""
        d = 1 - torch.einsum("nd,nkd->nk", s, self.P[0, y])
        return d.argmin(1)

    def cosdist(self, s, c):
        return torch.stack([1 - torch.einsum("nd,ckd->nck", v, self.P[u]) for u, v in enumerate((s, c))])  # 2,N,C,K

    @torch.no_grad()
    def ema_update(self, s, c, y):
        k = self.assign(s, y)
        d = self.cosdist(s, c)
        n = torch.arange(len(y), device=y.device)
        dd = d[:, n, y, k]  # 2,N
        for cl in y.unique():
            for j in range(self.K):
                m = (y == cl) & (k == j)
                if not m.any():
                    continue
                for u, v in enumerate((s, c)):
                    self.P[u, cl, j] = F.normalize(self.kappa * self.P[u, cl, j] + (1 - self.kappa) * v[m].mean(0), dim=0)
                    self.m1[u, cl, j] = self.kappa * self.m1[u, cl, j] + (1 - self.kappa) * dd[u, m].mean()
                    self.m2[u, cl, j] = self.kappa * self.m2[u, cl, j] + (1 - self.kappa) * (dd[u, m] ** 2).mean()

    @torch.no_grad()
    def recompute_stats(self, s, c, y):
        """exact moments on the full training set (used at init and after training)."""
        k = self.assign(s, y)
        d = self.cosdist(s, c)
        n = torch.arange(len(y), device=y.device)
        dd = d[:, n, y, k]
        if self.loo:  # leave-one-out member distances: prototype mean excludes the member itself
            for u, v in enumerate((s, c)):
                for cl in range(self.C):
                    for j in range(self.K):
                        m = (y == cl) & (k == j)
                        cnt = int(m.sum())
                        if cnt < 2:
                            continue
                        tot = v[m].sum(0, keepdim=True)
                        loo = F.normalize((tot - v[m]) / (cnt - 1), dim=1)
                        dd[u, m] = 1 - (v[m] * loo).sum(1)
        for cl in range(self.C):
            for j in range(self.K):
                m = (y == cl) & (k == j)
                self.cnt[cl, j] = float(m.sum())
                if int(m.sum()) < 2:
                    m = y == cl
                self.m1[:, cl, j] = dd[:, m].mean(1)
                self.m2[:, cl, j] = (dd[:, m] ** 2).mean(1)

    @torch.no_grad()
    def refresh(self, s, c, y, iters=3):
        """re-estimate prototypes (nearest-assignment refinement, fixed mode ids) and moments
        on the full training set in eval mode; keeps prototypes aligned with drifting features."""
        for _ in range(iters):
            k = self.assign(s, y)
            for cl in range(self.C):
                for j in range(self.K):
                    m = (y == cl) & (k == j)
                    if m.any():
                        self.P[0, cl, j] = F.normalize(s[m].mean(0), dim=0)
                        self.P[1, cl, j] = F.normalize(c[m].mean(0), dim=0)
        self.recompute_stats(s, c, y)

    def mu_sigma(self):
        m1, m2 = self.m1, self.m2
        if self.shrink > 0 and self.mode == "mode":
            # empirical-Bayes shrinkage of per-mode moments toward the class-pooled moments
            w = self.cnt[None]                                         # 1,C,K
            M1 = (w * m1).sum(2, keepdim=True) / w.sum(2, keepdim=True).clamp_min(1)
            M2 = (w * m2).sum(2, keepdim=True) / w.sum(2, keepdim=True).clamp_min(1)
            m1 = (w * m1 + self.shrink * M1) / (w + self.shrink)
            m2 = (w * m2 + self.shrink * M2) / (w + self.shrink)
        mu = m1
        sig = (m2 - m1 ** 2).clamp_min(0).sqrt()
        if self.mode == "class":
            mu = mu.mean(2, keepdim=True).expand_as(mu)
            sig = sig.mean(2, keepdim=True).expand_as(sig)
        elif self.mode == "global":
            mu = mu.mean((1, 2), keepdim=True).expand_as(mu)
            sig = sig.mean((1, 2), keepdim=True).expand_as(sig)
        return mu.detach(), sig.clamp_min(self.smin).detach()

    # ---- scoring
    def delta(self, s, c):
        """Eq. (9): 2,N,C normalised distance to each class."""
        d = self.cosdist(s, c)
        if self.mode == "abs":
            return d.min(-1).values
        mu, sig = self.mu_sigma()
        return ((d - mu[:, None]) / sig[:, None]).min(-1).values

    def score(self, s, c, rho, return_parts=False):
        dl = self.delta(s, c)
        q = (-dl / self.tau_a).softmax(-1)               # Eq. (10)
        ds, dc = dl[0].min(1).values, dl[1].min(1).values
        w = self.omega * rho if self.fixed_w is None else torch.full_like(rho, self.fixed_w)
        js = js_div(q[0], q[1])
        S = w * dc + (1 - w) * ds + self.gamma * js       # Eq. (11)
        if return_parts:
            return S, dict(delta=dl, q=q, js=js, w=w)
        return S
