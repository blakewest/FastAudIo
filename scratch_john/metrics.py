import torch
import numpy as np
from fastai.core import to_np


eps = np.finfo(float).eps

def mapk_np(preds, targs, k=3):
    preds = np.argsort(-preds, axis=1)[:, :k]
    score = 0.0
    for i in range(k):
        num_hits = (preds[:, i] == targs).sum()
        score += num_hits * (1.0 / (i+1.0))
    score /= preds.shape[0]
    return score

def mapk(preds, targs, k=3):
    return mapk_np(to_np(preds), to_np(targs), k)

def recall(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/(targs.sum() + eps)

def precision(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum()/(pred_pos.sum() + eps)

def fbeta(preds, targs, beta, thresh=0.5):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).

    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(preds, targs, thresh)
    prec = precision(preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)

def f1(preds, targs, thresh=0.5): return fbeta(preds, targs, 1, thresh)
