"""Data loading, open-set split and patch sampling for SCOPE-Net.

Raw (non-negative) reflectance is kept alongside the band-standardised cube:
the spectral-angle prior of CCE is computed on raw reflectance (Sec. 3.2),
the network input uses standardisation fitted on training pixels only.
"""
import os
import numpy as np
import scipy.io as sio
import torch

DATA_ROOT = os.environ.get("SCOPE_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))

# known / unknown protocol follows DSCP-Net (TGRS 2026) Tables I-IV
DATASETS = {
    "Houston": dict(img=r"Houston13\houston_hsi.mat", gt=r"Houston13\houston_gt.mat",
                    known=list(range(1, 13)), unknown=[13, 14, 15]),
    "IP": dict(img=r"Indian\Indian_pines_corrected.mat", gt=r"Indian\Indian_pines_gt.mat",
               known=list(range(1, 9)), unknown=list(range(9, 17))),
    "LK": dict(img=r"WHU_Hi_LongKou\WHU_Hi_LongKou.mat", gt=r"WHU_Hi_LongKou\WHU_Hi_LongKou_gt.mat",
               known=list(range(1, 9)), unknown=[9]),
    "UP": dict(img=r"PaviaU\PaviaU.mat", gt=r"PaviaU\PaviaU_gt.mat",
               known=list(range(1, 9)), unknown=[9]),
    # HyRank Dioni with classes 6 and 8 (absent from the scene) removed and relabelled 1-12 (Dioni_gt_out68);
    # known / unknown classes follow the HyRank protocol of the WGDT project (docs/PROTOCOL.md)
    "Dioni": dict(img=r"HyRank\Dioni.mat", gt=r"HyRank\Dioni_gt_out68.mat",
                  known=[1, 2, 3, 4, 5, 6, 8, 10, 12], unknown=[7, 9, 11]),
}


def _load_mat_array(path, ndim):
    d = sio.loadmat(path)
    cands = [v for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == ndim]
    if ndim == 2:  # pick the label map (integer, largest), skip colour maps
        cands = [v for v in cands if np.issubdtype(v.dtype, np.integer) or v.shape[1] > 3]
    if not cands:
        raise ValueError(f"no {ndim}-D array in {path}")
    return max(cands, key=lambda v: v.size)


def load_scene(name, root=DATA_ROOT):
    cfg = DATASETS[name]
    img = _load_mat_array(os.path.join(root, cfg["img"]), 3).astype(np.float32)
    gt = _load_mat_array(os.path.join(root, cfg["gt"]), 2).astype(np.int64)
    assert img.shape[:2] == gt.shape, (img.shape, gt.shape)
    img = img - min(0.0, float(img.min()))  # guarantee non-negative raw reflectance
    return img, gt, cfg


def make_split(gt, known, unknown, seed, n_pool=20, n_val=5, fold=0):
    """Per known class: n_pool labelled pixels -> (n_pool-n_val) train + n_val val.
    Remaining known pixels and all unknown pixels form the test set.
    fold k (cross-fitting) uses the k-th block of n_val labelled pixels as validation and the rest
    for training; fold 0 is the default split and the test set does not depend on the fold."""
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in known:
        idx = np.flatnonzero(gt.ravel() == c)
        rng.shuffle(idx)
        k = min(n_pool, max(len(idx) - 1, 2))
        nv = min(n_val, k // 2)
        f = fold % max(1, k // nv)
        va.append(idx[f * nv:(f + 1) * nv]); tr.append(np.concatenate([idx[:f * nv], idx[(f + 1) * nv:k]])); te.append(idx[k:])
    for c in unknown:
        te.append(np.flatnonzero(gt.ravel() == c))
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


class Scene:
    """Holds padded cubes on a device and serves patches by flat pixel index."""

    def __init__(self, name, seed, patch=9, device="cuda", root=DATA_ROOT, n_pool=20, n_val=5, fold=0):
        img, gt, cfg = load_scene(name, root)
        self.name, self.H, self.W, self.B = name, *img.shape
        self.known, self.unknown = cfg["known"], cfg["unknown"]
        self.C = len(self.known)
        self.gt = gt
        # label mapping: known -> 0..C-1, unknown -> C, background -> -1
        lab = -np.ones_like(gt)
        for i, c in enumerate(self.known):
            lab[gt == c] = i
        for c in self.unknown:
            lab[gt == c] = self.C
        self.lab = lab
        self.train_idx, self.val_idx, self.test_idx = make_split(gt, self.known, self.unknown, seed, n_pool, n_val, fold)
        # standardisation fitted on training pixels only
        flat = img.reshape(-1, self.B)
        mu = flat[self.train_idx].mean(0)
        sd = flat[self.train_idx].std(0) + 1e-6
        std = (img - mu) / sd
        self.p, r = patch, patch // 2
        pad = lambda a: np.pad(a, ((r, r), (r, r), (0, 0)), mode="reflect")
        self.device = device
        self.x = torch.from_numpy(pad(std).transpose(2, 0, 1).copy()).to(device)   # B,H+2r,W+2r
        self.raw = torch.from_numpy(pad(img).transpose(2, 0, 1).copy()).to(device)
        self.labels = torch.from_numpy(lab.ravel()).to(device)
        off = torch.arange(patch, device=device)
        self._oy, self._ox = torch.meshgrid(off, off, indexing="ij")

    def patches(self, idx):
        """idx: LongTensor of flat pixel indices -> (x[N,B,P,P], raw[N,B,P,P], y[N])."""
        idx = idx.to(self.device)
        rows, cols = idx // self.W, idx % self.W
        R = rows[:, None, None] + self._oy  # N,P,P (padded coords: top-left = row)
        Cc = cols[:, None, None] + self._ox
        x = self.x[:, R, Cc].permute(1, 0, 2, 3)
        raw = self.raw[:, R, Cc].permute(1, 0, 2, 3)
        return x, raw, self.labels[idx]

    def class_pool(self, idx):
        """dict class -> LongTensor of flat indices (for SSR partner sampling)."""
        y = self.labels[torch.as_tensor(idx, device=self.device)]
        idx = torch.as_tensor(idx, device=self.device)
        return {c: idx[y == c] for c in range(self.C)}
