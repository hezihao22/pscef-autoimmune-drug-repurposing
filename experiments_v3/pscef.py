"""Positive support calibrated evidence fusion.

This module receives expert scores and positive support row indices only.
It deliberately has no query-label argument.
"""
import numpy as np
from scipy.stats import rankdata

def fuse_scores(expert_scores, positive_support_indices, temperature=4.0, weight_scale=1.0):
    """Return candidate scores, expert weights, and calibration intermediates.

    expert_scores: finite array (candidate_count, expert_count), larger is better.
    positive_support_indices: unique row indices of verified known positives.
    The returned scores are ranking values, not clinical probabilities.
    """
    scores=np.asarray(expert_scores,dtype=float)
    raw_support=np.asarray(positive_support_indices)
    if not np.issubdtype(raw_support.dtype,np.integer):
        raise ValueError('Positive support indices must be integers.')
    support=raw_support.astype(int,copy=False)
    if scores.ndim!=2 or min(scores.shape)<1 or not np.isfinite(scores).all():
        raise ValueError('Expert scores must be a nonempty finite 2D array.')
    if support.ndim!=1 or not len(support) or len(np.unique(support))!=len(support):
        raise ValueError('Provide at least one unique positive support index.')
    if support.min()<0 or support.max()>=len(scores):
        raise ValueError('Positive support index outside candidate array.')
    if not np.isfinite(temperature) or not np.isfinite(weight_scale) or temperature<0 or weight_scale<0:
        raise ValueError('Temperature and weight scale must be finite and nonnegative.')
    n=len(scores)
    ranks=np.column_stack([(rankdata(scores[:,m],method='average')-.5)/n for m in range(scores.shape[1])])
    evidence=-np.log(np.maximum(1-ranks[support],1/(2*n))).mean(axis=0)
    weights=np.exp(weight_scale*(evidence-evidence.max()));weights/=weights.sum()
    fused=np.exp(temperature*(ranks-1))@weights
    return {'scores':fused,'weights':weights,'percentiles':ranks,'positive_tail_evidence':evidence}
