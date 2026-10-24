"""Graph family for the generator: two-hop user -> item -> item relations.

Uses only the schema roles (user, item, time) of the interaction table and history
strictly before the cutoff. Every value is a generic aggregate over the bipartite
user-item graph, so no domain knowledge enters: co-visitation of the candidate with
the user's history (sum with time decay, max, mean, last-k), the same on a recent
window, and a low-rank (SVD) user-item affinity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from ..common.contracts import historical_events


def _cosine(matrix):
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    cooc = (matrix.T @ matrix).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    inv = sparse.diags(1 / np.sqrt(np.maximum(counts, 1)))
    return (inv @ cooc @ inv).tocsr().astype(np.float32)


class GraphFamily:
    def __init__(self, half_lives=(7, 30), last_k=(1, 3, 10), recent_days=14, svd_rank=64, seed=0,
                 user='user_id', item='item_id', time='last_watch_dt'):
        self.half_lives, self.last_k = tuple(half_lives), tuple(last_k)
        self.recent_days, self.svd_rank, self.seed = recent_days, svd_rank, seed
        self.user, self.item, self.time = user, item, time

    @property
    def columns(self):
        cols = [f'g__covis_decay{h}' for h in self.half_lives]
        cols += [f'g__covis_decay{h}_norm' for h in self.half_lives]
        cols += ['g__covis_max', 'g__covis_mean', 'g__covis_sum']
        cols += [f'g__covis_last{k}' for k in self.last_k]
        cols += [f'g__recent{self.recent_days}_covis_sum', f'g__recent{self.recent_days}_covis_max',
                 f'g__recent{self.recent_days}_covis_last3']
        cols += [f'g__svd{self.svd_rank}_dot', f'g__svd{self.svd_rank}_cos']
        return cols

    def transform(self, events, cutoff, keys):
        u, i, t = self.user, self.item, self.time
        h = historical_events(events, cutoff)
        users, items = np.sort(h[u].unique()), np.sort(h[i].unique())
        ui = pd.Index(users).get_indexer(h[u])
        ii = pd.Index(items).get_indexer(h[i])
        shape = (len(users), len(items))
        X = sparse.csr_matrix((np.ones(len(h), dtype=np.float32), (ui, ii)), shape=shape)
        S = _cosine(X)
        recent = (h[t] >= cutoff - pd.Timedelta(days=self.recent_days)).to_numpy()
        Xr = sparse.csr_matrix((np.ones(recent.sum(), dtype=np.float32), (ui[recent], ii[recent])),
                               shape=shape)
        R = _cosine(Xr)
        # Low-rank affinity on an idf-weighted matrix; users folded in through their rows.
        idf = np.log1p(len(users) / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1)).astype(np.float32)
        Xw = (X @ sparse.diags(idf)).tocsr()
        rank = min(self.svd_rank, min(shape) - 1)
        _, _, vt = randomized_svd(Xw, rank, random_state=self.seed)
        item_vec = vt.T.astype(np.float32)
        user_vec = np.asarray(Xw @ item_vec)

        age = ((cutoff - h[t]).dt.total_seconds() / 86400).to_numpy()
        hist = pd.DataFrame({'u': ui, 'i': ii, 'age': age}).sort_values(['u', 'age', 'i'])
        hist_ptr = np.searchsorted(hist.u.to_numpy(), np.arange(len(users) + 1))
        h_items, h_age = hist.i.to_numpy(), hist.age.to_numpy()

        q = pd.DataFrame({'u': pd.Index(users).get_indexer(keys[u]),
                          'i': pd.Index(items).get_indexer(keys[i]), 'row': np.arange(len(keys))})
        out = np.full((len(keys), len(self.columns)), np.nan, dtype=np.float32)
        q = q.loc[(q.u >= 0) & (q.i >= 0)].sort_values('u')
        q_u, q_i, q_row = q.u.to_numpy(), q.i.to_numpy(), q.row.to_numpy()
        bounds = np.flatnonzero(np.r_[True, q_u[1:] != q_u[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            row = q_u[a]
            lo, hi = hist_ptr[row], hist_ptr[row + 1]
            hi_items, ages = h_items[lo:hi], h_age[lo:hi]
            cand = q_i[a:b]
            M = S[hi_items][:, cand].toarray()
            Mr = R[hi_items][:, cand].toarray()
            feats = []
            weights = [np.exp2(-ages / hl) for hl in self.half_lives]
            feats += [w @ M for w in weights]
            feats += [(w @ M) / max(w.sum(), 1e-9) for w in weights]
            feats += [M.max(axis=0), M.mean(axis=0), M.sum(axis=0)]
            feats += [M[:k].mean(axis=0) for k in self.last_k]
            feats += [Mr.sum(axis=0), Mr.max(axis=0), Mr[:3].mean(axis=0)]
            uv, iv = user_vec[row], item_vec[cand]
            dot = iv @ uv
            norm = np.linalg.norm(iv, axis=1) * np.linalg.norm(uv)
            feats += [dot, np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)]
            out[q_row[a:b]] = np.stack(feats, axis=1)
        return pd.DataFrame(out, columns=self.columns)


class FilteredPopularity:
    """Item popularity under schema-derived row filters and window ratios.

    Filters come from the event table only: a completion threshold on the numeric
    measure, and the audience of users active in the last week.
    """

    def __init__(self, windows=(1, 2, 3, 5, 7, 10, 14, 21), measure='watched_pct', threshold=50.,
                 user='user_id', item='item_id', time='last_watch_dt'):
        self.windows, self.measure, self.threshold = tuple(windows), measure, threshold
        self.user, self.item, self.time = user, item, time

    @property
    def columns(self):
        cols = [f'p__count{w}' for w in self.windows]
        cols += [f'p__complete_count{w}' for w in (3, 7, 14)]
        cols += [f'p__active_audience_count{w}' for w in (7, 30)]
        cols += ['p__ratio_1_7', 'p__ratio_3_14', 'p__ratio_7_21', 'p__first_seen_days',
                 'p__complete_share7']
        return cols

    def transform(self, events, cutoff, keys):
        u, i, t = self.user, self.item, self.time
        h = historical_events(events, cutoff)
        age = (cutoff - h[t]).dt.total_seconds() / 86400
        item = keys[i]
        out = {}
        def count(mask):
            return item.map(h.loc[mask].groupby(i).size()).fillna(0).to_numpy(dtype=np.float32)
        for w in self.windows:
            out[f'p__count{w}'] = count(age < w)
        complete = h[self.measure] >= self.threshold
        for w in (3, 7, 14):
            out[f'p__complete_count{w}'] = count((age < w) & complete)
        active = h.loc[age < 7, u].unique()
        for w in (7, 30):
            out[f'p__active_audience_count{w}'] = count((age < w) & h[u].isin(active))
        out['p__ratio_1_7'] = (out['p__count1'] + 1) / (out['p__count7'] / 7 + 1)
        out['p__ratio_3_14'] = (out['p__count3'] / 3 + 1) / (out['p__count14'] / 14 + 1)
        out['p__ratio_7_21'] = (out['p__count7'] / 7 + 1) / (out['p__count21'] / 21 + 1)
        out['p__first_seen_days'] = item.map(age.groupby(h[i]).max()).to_numpy(dtype=np.float32)
        out['p__complete_share7'] = (out['p__complete_count7'] + 1) / (out['p__count7'] + 2)
        return pd.DataFrame(out)[self.columns].astype(np.float32)


def families(graph, pop):
    """Candidate families for the greedy V4 selection on top of V2."""
    g = graph.columns
    return {'covis': [c for c in g if c.startswith('g__covis')],
            'recent_covis': [c for c in g if c.startswith('g__recent')],
            'svd': [c for c in g if c.startswith('g__svd')],
            'popularity': list(pop.columns)}
