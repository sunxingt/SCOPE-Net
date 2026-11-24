"""Open-set metrics: OA, OS* (known-class mean accuracy), per-class acc, UNK, Kappa, HOS (Eq. 21), HOS_OA, AUROC."""
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def open_set_metrics(y_true, y_pred, score, C):
    """y_true/y_pred in {0..C-1, C=unknown}; score = unknown score (higher = more unknown)."""
    y_true, y_pred, score = map(np.asarray, (y_true, y_pred, score))
    per_cls = np.array([(y_pred[y_true == c] == c).mean() if (y_true == c).any() else np.nan for c in range(C)])
    unk_mask = y_true == C
    unk = float((y_pred[unk_mask] == C).mean()) if unk_mask.any() else float("nan")
    oa = float((y_pred == y_true).mean())
    os_star = float(np.nanmean(per_cls))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    hos = 2 * os_star * unk / (os_star + unk + 1e-12)
    hos_oa = 2 * oa * unk / (oa + unk + 1e-12)
    try:
        auroc = float(roc_auc_score(unk_mask.astype(int), score))
    except ValueError:
        auroc = float("nan")
    return dict(OA=oa, OS_star=os_star, UNK=unk, Kappa=kappa, HOS=hos, HOS_OA=hos_oa, AUROC=auroc,
                per_class=per_cls.tolist())
