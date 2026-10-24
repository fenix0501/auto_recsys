"""Temporal popularity and sparse cosine ItemKNN, with no target injection."""
from __future__ import annotations
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from .contracts import historical_events

COMMON_FEATURES = ['popular_rank', 'knn_score', 'knn_rank', 'source_knn']


def build_candidates(events, cutoff, user_ids, config):
    start = perf_counter()
    history = historical_events(events, cutoff)
    catalog = np.sort(history['item_id'].unique())
    warm_users = np.sort(history['user_id'].unique())
    if not len(catalog):
        raise ValueError('No historical items at cutoff')
    ui = pd.Index(warm_users).get_indexer(history['user_id'])
    ii = pd.Index(catalog).get_indexer(history['item_id'])
    matrix = sparse.csr_matrix((np.ones(len(history), dtype=np.float32), (ui, ii)),
                              shape=(len(warm_users), len(catalog)))
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    cooc = (matrix.T @ matrix).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    inv = 1 / np.sqrt(np.maximum(counts, 1))
    similarity = (sparse.diags(inv) @ cooc @ sparse.diags(inv)).tocsr()
    # Keep only the strongest fixed number of neighbors per historical item.
    values, columns, pointers = [], [], [0]
    for i in range(len(catalog)):
        lo, hi = similarity.indptr[i:i+2]
        cols, vals = similarity.indices[lo:hi], similarity.data[lo:hi]
        order = np.lexsort((catalog[cols], -vals))[:config['knn_neighbors']]
        values.extend(vals[order])
        columns.extend(cols[order])
        pointers.append(len(values))
    similarity = sparse.csr_matrix((np.array(values, dtype=np.float32),
                                    np.array(columns, dtype=np.int32),
                                    np.array(pointers, dtype=np.int32)),
                                   shape=(len(catalog), len(catalog)))
    scores = (matrix @ similarity).tocsr()
    recent = history[history['last_watch_dt'] >= cutoff-pd.Timedelta(days=7)]
    pop7 = recent.groupby('item_id').size().reindex(catalog, fill_value=0).to_numpy()
    pop_order = np.lexsort((catalog, -counts, -pop7))
    pop_rank = np.empty(len(catalog), dtype=np.int32)
    pop_rank[pop_order] = np.arange(1, len(catalog)+1)
    user_lookup = {int(u): i for i, u in enumerate(warm_users)}
    rows = []
    for user in sorted(map(int, user_ids)):
        if user not in user_lookup:
            continue
        row = user_lookup[user]
        seen = set(matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]])
        score_row = scores.getrow(row)
        usable = np.array([c not in seen for c in score_row.indices], dtype=bool)
        neighbors, vals = score_row.indices[usable], score_row.data[usable]
        order = np.lexsort((catalog[neighbors], -vals))
        neighbors, vals = neighbors[order], vals[order]
        knn_top = neighbors[:config['knn_candidates']]
        knn_set = set(knn_top)
        score_map = dict(zip(neighbors, vals))
        rank_map = {int(c): r+1 for r, c in enumerate(neighbors)}
        available_pop = [int(c) for c in pop_order if c not in seen]
        popular_n = config['candidate_limit'] - config['knn_candidates']
        chosen = list(dict.fromkeys([*map(int, knn_top), *available_pop[:popular_n]]))
        chosen_set = set(chosen)
        for c in available_pop:
            if len(chosen) >= config['candidate_limit']:
                break
            if c not in chosen_set:
                chosen.append(c)
                chosen_set.add(c)
        for c in chosen:
            rows.append((user, int(catalog[c]), int(pop_rank[c]),
                         float(score_map.get(c, 0)), rank_map.get(c, len(catalog)+1),
                         int(c in knn_set)))
    candidates = pd.DataFrame(rows, columns=['user_id', 'item_id', *COMMON_FEATURES])
    candidates = candidates.sort_values(['user_id', 'item_id']).reset_index(drop=True)
    for c in candidates.columns:
        candidates[c] = candidates[c].astype('float32' if c == 'knn_score' else 'int32')
    diagnostics = {'candidate_seconds': perf_counter()-start, 'history_rows': len(history),
                   'catalog_items': len(catalog), 'candidate_rows': len(candidates),
                   'candidate_users': candidates['user_id'].nunique(),
                   'similarity_nonzero': similarity.nnz}
    return candidates, diagnostics
