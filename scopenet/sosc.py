"""SOSC: spectral segment recombination (Eq. 15), cross-class embedding interpolation (Eq. 16),
open-space losses (Eqs. 17-18) and pseudo-open calibration (Eq. 19)."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def nearest_class_map(bank):
    """class a -> nearest other known class by closest signature-space prototype pair."""
    P = bank.P[0]                                   # C,K,D
    C = P.shape[0]
    sim = torch.einsum("akd,bld->abkl", P, P).amax((2, 3))
    sim.fill_diagonal_(-2)
    return sim.argmax(1)                            # C


def band_mask(n, B, L, rmin, rmax, nt, gen, device):
    """n masks in [0,1]^B: L equal segments, ratio r of them replaced, nt-band linear transitions."""
    edges = np.linspace(0, B, L + 1).round().astype(int)
    m = torch.zeros(n, B, device=device)
    for i in range(n):
        r = rmin + (rmax - rmin) * torch.rand(1, generator=gen).item()
        k = max(1, int(round(r * L)))
        segs = torch.randperm(L, generator=gen)[:k].tolist()
        for sgm in segs:
            m[i, edges[sgm]:edges[sgm + 1]] = 1.0
    if nt > 0:  # linear transition = box smoothing of width 2*nt+1
        ker = torch.ones(1, 1, 2 * nt + 1, device=device) / (2 * nt + 1)
        m = F.conv1d(F.pad(m[:, None], (nt, nt), mode="replicate"), ker)[:, 0]
    return m


def sample_partners(y, nmap, pools, pairing, C, gen):
    """for each label a pick a flat index from class nmap[a] (or a random other class)."""
    out = []
    for a in y.tolist():
        b = int(nmap[a]) if pairing == "nearest" else int(torch.randint(C - 1, (1,), generator=gen))
        if pairing != "nearest" and b >= a:
            b += 1
        pool = pools[b]
        out.append(pool[torch.randint(len(pool), (1,), generator=gen).item()])
    return torch.stack(out)


def ssr(scene, idx_a, y_a, nmap, pools, cfg, gen):
    """Eq. (15) on both standardised and raw cubes (standardisation is per-band affine,
    so mixing either representation with the same mask is equivalent)."""
    xa, ra, _ = scene.patches(idx_a)
    idx_b = sample_partners(y_a, nmap, pools, cfg.pairing, scene.C, gen)
    xb, rb, _ = scene.patches(idx_b)
    m = band_mask(len(idx_a), scene.B, cfg.L, cfg.rmin, cfg.rmax, cfg.nt, gen, xa.device)[:, :, None, None]
    return (1 - m) * xa + m * xb, (1 - m) * ra + m * rb


def fmi(s, c, rho, y, nmap, pairing, gen, return_idx=False):
    """Eq. (16): spherical interpolation between batch embeddings of class a and its nearest class."""
    n = len(y)
    j_idx = []
    for i in range(n):
        target = int(nmap[y[i]])
        cand = torch.nonzero(y == target).flatten() if pairing == "nearest" else torch.empty(0, dtype=torch.long)
        if len(cand) == 0:
            cand = torch.nonzero(y != y[i]).flatten()
        if len(cand) == 0:
            j_idx.append(-1); continue
        j_idx.append(int(cand[torch.randint(len(cand), (1,), generator=gen)]))
    j = torch.tensor(j_idx, device=y.device)
    keep = j >= 0
    i = torch.arange(n, device=y.device)[keep]; j = j[keep]
    t = (0.35 + 0.3 * torch.rand(len(i), generator=gen)).to(s.device)[:, None]
    zs = F.normalize(t * s[i] + (1 - t) * s[j], dim=1)
    zc = F.normalize(t * c[i] + (1 - t) * c[j], dim=1)
    zr = t[:, 0] * rho[i] + (1 - t[:, 0]) * rho[j]
    if return_idx:
        return zs, zc, zr, i
    return zs, zc, zr


def loss_open(S_pseudo, m_out):
    return F.relu(m_out - S_pseudo).mean() if S_pseudo.numel() else S_pseudo.new_zeros(())


def loss_uniform(logits):
    return -logits.log_softmax(1).mean()


def calibrate(S_known, S_pseudo, alpha):
    """Eq. (19). Returns theta*, lower bound and the objective value."""
    S_known = np.sort(np.asarray(S_known)); S_pseudo = np.asarray(S_pseudo)
    lb = float(np.quantile(S_known, 1 - alpha, method="higher"))
    cands = np.unique(np.concatenate([S_known, S_pseudo, [lb]]))
    cands = cands[cands >= lb]
    tpr = np.searchsorted(S_known, cands, side="right") / len(S_known)
    tnr = np.array([(S_pseudo > t).mean() for t in cands]) if len(S_pseudo) else np.zeros_like(cands)
    J = 0.5 * (tpr + tnr)
    best = np.flatnonzero(np.isclose(J, J.max()))
    lo, hi = cands[best[0]], cands[best[-1]]
    # optimal interval is [cands[best[0]], next candidate after best[-1]) -> take midpoint
    nxt = cands[best[-1] + 1] if best[-1] + 1 < len(cands) else hi
    theta = 0.5 * (lo + nxt) if nxt > lo else lo
    return float(theta), lb, float(J.max())


def quantile_threshold(S_known, q):
    return float(np.quantile(np.asarray(S_known), q))
