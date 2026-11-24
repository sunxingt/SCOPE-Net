"""Scene-level masked spectral pretraining of the pixel-feature backbone (no labels used).

Patches centred at randomly drawn scene pixels are masked by dropping a random subset of contiguous
band groups (the same groups for every pixel of a patch). The backbone PixelFeatures maps the masked
patch to pixel features, and a light decoder reconstructs the standardized spectrum of every pixel;
the loss is the mean squared error on the masked bands only. The pretrained backbone initializes the
encoder before the few-shot open-set training.
"""
import time
import numpy as np
import torch
import torch.nn as nn


def masked_spectral_pretrain(model, scene, args, gen, log=print):
    pix = model.enc.pix
    B, D = scene.B, args.dim
    dec = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, B)).to(scene.device)
    params = list(pix.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.pre_lr, weight_decay=1e-4)
    groups = torch.arange(B, device=scene.device) * args.pre_groups // B        # band -> group id
    if getattr(args, "pre_pool", "all") == "labelfree":
        # control: centres whose patch window holds no labelled pixel, so no training, validation or test
        # pixel enters any input or reconstruction target during pretraining
        from scipy.ndimage import maximum_filter
        near = maximum_filter((scene.gt > 0).astype(np.uint8), size=scene.p, mode="constant") > 0
        pool = torch.as_tensor(np.flatnonzero(~near.ravel()), device=scene.device)
        assert len(pool) >= 1000, f"only {len(pool)} label-free windows in {scene.name}"
    else:
        pool = torch.arange(scene.H * scene.W, device=scene.device)
    t0 = time.time(); hist = []
    pix.train(); dec.train()
    for ep in range(args.pre_ep):
        idx = pool[torch.randint(0, len(pool), (args.pre_n,), generator=gen).to(scene.device)]
        tot = 0.0
        for i in range(0, len(idx), args.pre_bs):
            x, _, _ = scene.patches(idx[i:i + args.pre_bs])                    # N,B,P,P (standardized)
            n = x.shape[0]
            gm = torch.rand(n, args.pre_groups, generator=gen).to(scene.device) < args.pre_mask
            M = gm[:, groups]                                                   # N,B masked bands
            Fp = pix(x * (~M)[:, :, None, None].float())                       # N,P*P,D
            rec = dec(Fp)                                                       # N,P*P,B
            tgt = x.flatten(2).transpose(1, 2)                                  # N,P*P,B
            m = M[:, None, :].expand_as(rec).float()
            loss = (((rec - tgt) ** 2) * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * n
        hist.append(round(tot / len(idx), 4))
        if ep % 10 == 0 or ep == args.pre_ep - 1:
            log(f"[{args.dataset} s{args.seed} {args.tag}] pretrain ep {ep:3d} masked-MSE {hist[-1]:.4f} ({time.time() - t0:.0f}s)")
    return dict(pretrain_loss=hist, pretrain_time=time.time() - t0)
