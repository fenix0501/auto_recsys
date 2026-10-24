"""Macro user metrics with complete target sets, including missed candidates."""
from __future__ import annotations
import numpy as np
import pandas as pd


def attach_labels(candidates, targets):
    out = candidates[candidates['user_id'].isin(targets)].copy().reset_index(drop=True)
    out['label'] = np.fromiter((int(i in targets[int(u)])
                               for u, i in zip(out['user_id'], out['item_id'])),
                              dtype=np.int8, count=len(out))
    return out


class RankingEvaluator:
    """Precompute fixed labels, full ideal DCG and user mappings once."""
    def __init__(self, frame, targets, k=10):
        self.k = k
        self.users = np.array(sorted(targets), dtype=np.int64)
        self.item_ids = frame['item_id'].to_numpy()
        self.frame_users = frame['user_id'].to_numpy()
        if not set(self.frame_users).issubset(targets):
            raise ValueError('Candidates include users outside evaluation population')
        self.user_index = np.searchsorted(self.users, self.frame_users)
        self.labels = np.fromiter((int(i in targets[int(u)]) for u, i in
                                   zip(self.frame_users, self.item_ids)),
                                  dtype=np.int8, count=len(frame))
        self.n_true = np.array([len(targets[int(u)]) for u in self.users], dtype=float)
        if not len(self.users) or (self.n_true == 0).any():
            raise ValueError('Evaluation requires nonempty target sets')
        discounts = 1 / np.log2(np.arange(k)+2)
        ideal_prefix = np.r_[0., np.cumsum(discounts)]
        self.ideal = ideal_prefix[np.minimum(self.n_true.astype(int), k)]
        self.candidate_hits = np.bincount(self.user_index, weights=self.labels,
                                          minlength=len(self.users))

    def per_user(self, scores):
        scores = np.asarray(scores)
        if len(scores) != len(self.labels) or not np.isfinite(scores).all():
            raise ValueError('Invalid scores')
        order = np.lexsort((self.item_ids, -scores, self.frame_users))
        group = self.user_index[order]
        labels = self.labels[order]
        if len(group):
            starts = np.r_[True, group[1:] != group[:-1]]
            start_positions = np.maximum.accumulate(np.where(starts, np.arange(len(group)), 0))
            rank = np.arange(len(group)) - start_positions
            top = rank < self.k
            dcg = np.bincount(group[top], weights=labels[top] / np.log2(rank[top]+2),
                               minlength=len(self.users))
            hits = np.bincount(group[top], weights=labels[top], minlength=len(self.users))
        else:
            dcg = hits = np.zeros(len(self.users))
        return pd.DataFrame({'user_id': self.users, 'ndcg10': dcg / self.ideal,
                             'recall10': hits / self.n_true,
                             'candidate_recall': self.candidate_hits / self.n_true,
                             'target_items': self.n_true.astype(int)})

    def summary(self, scores):
        values = self.per_user(scores)
        return {c: float(values[c].mean()) for c in ['ndcg10', 'recall10', 'candidate_recall']}

    def lightgbm_metric(self, y_true, y_pred):
        # Built-in NDCG normalizes only by candidate positives and can score empty
        # positive groups as perfect. Early stopping must match our actual task.
        return 'full_ndcg10', self.summary(y_pred)['ndcg10'], True


def paired_bootstrap(left, right, repeats=500, seed=42):
    merged = left[['user_id', 'ndcg10']].merge(right[['user_id', 'ndcg10']],
              on='user_id', suffixes=('_left', '_right'), validate='one_to_one')
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError('Bootstrap populations differ')
    delta = (merged['ndcg10_left'] - merged['ndcg10_right']).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {'delta_ndcg10': float(delta.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi),
            'users': len(delta), 'repeats': repeats,
            'interpretation': 'paired user bootstrap, conditional on one split/sample/trained model'}
