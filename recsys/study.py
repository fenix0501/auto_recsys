"""Unified study: the manual pipeline and generators V1-V4 on one user split and one set of weeks.

Protocol (declared in protocol.json before any data is loaded):
  * users: a seeded permutation (seed 2026, blocks of 20 000, eligibility = history before
    the first train-week target, users of earlier development runs excluded);
    train = block 1, validation = blocks 2+3, test = blocks 6+7 (never opened before;
    blocks 0, 4, 5 are not used). Each block is processed as its own population, so
    candidate and graph statistics match the 20 000-user training population;
  * weeks: training cutoffs 12.07, 19.07, 26.07, 02.08 on train users (target = next 7 days),
    validation cutoff 09.08, test cutoff 16.08;
  * every generator (V1, V2, V3, V4) is fit on train-user history before 12.07; every choice
    (V2 windows/families, V3 formula subset, V4 families, early stopping, Platt scaling,
    manual ablations) is made on validation only;
  * plan.json with frozen feature sets and declared contrasts is written before the test
    users are loaded; the test is opened once after all models are saved.

Pipelines can be run separately (--pipelines manual v2 ...): every fit is deterministic and
independent of the other pipelines, so a subset reproduces the same numbers as the full run.
V4 is built on top of the selected V2, so it always runs the V2 search.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .autogen.v1_schema import SETTINGS as V1_SETTINGS, FeatureGenerator, prune_train
from .autogen.v2_grammar import VARIANTS, FeatureGeneratorV2
from .autogen.v3_formulas import FormulaGenerator, attribute_lift, window
from .autogen.v4_graph import FilteredPopularity, GraphFamily, families as graph_families
from .common.candidates import COMMON_FEATURES, build_candidates
from .common.data import ROOT, load_raw, log, split_dates, target_map, user_blocks, write_json
from .common.metrics import RankingEvaluator, attach_labels
from .manual.features import CATEGORICAL_FEATURES, CORE_FEATURES, MANUAL_FEATURES, build_features
from .manual.groups import INDIRECT, PRIMARY

BLOCKS = {'train': [1], 'validation': [2, 3], 'test': [6, 7]}
UNUSED_BLOCKS = [0, 4, 5]
TRAIN_CUTOFFS = ['2021-07-12', '2021-07-19', '2021-07-26', '2021-08-02']
V1_WINDOWS = (7, 30)
V2_WINDOWS = [(3, 30), (7, 30), (10, 30), (14, 30)]
V3_PREFIXES = [24, 48, 80, 120]
V3_PROBE_TREES = 80
V4_THRESHOLD = .8
PIPELINES = ['manual', 'v1', 'v2', 'v3', 'v4']
CONTRASTS = [('auto_v2+scores', 'manual+scores'), ('auto_v2', 'manual'),
             ('auto_v3+scores', 'manual+scores'), ('auto_v4+scores', 'auto_v2+scores'),
             ('manual_auto_v2+scores', 'manual+scores'), ('manual+scores', 'manual'),
             ('manual', 'popular'), ('auto_v1+scores', 'manual+scores'),
             # manual set without the item_behavior group (the best drop-one ablation)
             ('auto_v2+scores', 'manual_noib+scores'), ('auto_v2', 'manual_noib'),
             ('manual_noib+scores', 'manual+scores'), ('auto_v4+scores', 'manual_noib+scores')]
METRICS = ['ndcg10', 'recall10', 'roc_auc', 'logloss']

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*eval_set.*')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


def assign_blocks(events, cutoff, excluded, size, seed):
    """Blocks of the shared permutation; checks completeness and disjointness."""
    used = sorted(b for ids in BLOCKS.values() for b in ids)
    blocks = user_blocks(events, cutoff, excluded, size, max(used) + 1, seed)
    for b in used:
        if len(blocks[b]) != size:
            raise ValueError(f'Block {b} has {len(blocks[b])} users, expected {size}')
        assert not np.isin(blocks[b], excluded).any()
    for i, a in enumerate(used):
        for b in used[i + 1:]:
            assert not np.intersect1d(blocks[a], blocks[b]).size, f'blocks {a} and {b} overlap'
    return blocks, {split: [blocks[b] for b in ids] for split, ids in BLOCKS.items()}


class Part:
    """One population (block) at one cutoff: candidates, manual features and labels."""

    def __init__(self, events, users, items, cutoff, end, population, config, split, block):
        self.events, self.cutoff, self.split, self.block = events, cutoff, split, block
        self.targets = target_map(events, cutoff, end)
        tick = perf_counter()
        pairs, _ = build_candidates(events, cutoff, population, config)
        pairs = pairs.loc[pairs.user_id.isin(self.targets)].reset_index(drop=True)
        self.candidates_s = perf_counter() - tick
        tick = perf_counter()
        manual, _, _ = build_features(events, cutoff, pairs, users, items, config)
        self.manual_s = perf_counter() - tick
        frame = attach_labels(manual[['user_id', 'item_id', *COMMON_FEATURES, *MANUAL_FEATURES]], self.targets)
        for c in [*COMMON_FEATURES, *MANUAL_FEATURES]:
            frame[c] = frame[c].astype('float32')
        frame['week'] = str(cutoff.date())
        frame['block'] = block
        self.frame = frame
        self.keys = frame[['user_id', 'item_id']]
        log(f'{split} block {block} {cutoff.date()}: {len(self.targets)} users, {len(frame)} pairs, '
            f'cand {self.candidates_s:.1f}s manual {self.manual_s:.1f}s')


class Split:
    """Parts concatenated in (week, user, item) order; generated columns follow that order."""

    def __init__(self, name, parts):
        self.name, self.parts = name, parts
        base = pd.concat([p.frame for p in parts], ignore_index=True)
        self.order = np.lexsort((base.item_id.to_numpy(), base.user_id.to_numpy(), base.week.to_numpy()))
        self.frame = base.iloc[self.order].reset_index(drop=True)
        self.timings = {'candidates': sum(p.candidates_s for p in parts), 'manual': sum(p.manual_s for p in parts)}
        if len({p.cutoff for p in parts}) == 1:
            self.targets = {}
            for p in parts:
                assert not set(p.targets) & set(self.targets)
                self.targets.update(p.targets)
            self.evaluator = RankingEvaluator(self.frame, self.targets, 10)
            self.groups = self.frame.groupby('user_id', sort=True).size().to_numpy()
        g = self.frame.groupby(['week', 'user_id'], sort=False)['label'].agg(['sum', 'size'])
        keep = g.index[(g['sum'] > 0) & (g['sum'] < g['size'])]
        mask = pd.MultiIndex.from_frame(self.frame[['week', 'user_id']]).isin(keep)
        self.rank_rows = np.flatnonzero(mask)
        self.rank_groups = self.frame.iloc[self.rank_rows].groupby(['week', 'user_id'], sort=False).size().to_numpy()

    def compute(self, fn, label=None):
        tick = perf_counter()
        values = pd.concat([fn(p).reset_index(drop=True) for p in self.parts], ignore_index=True)
        values = values.iloc[self.order].reset_index(drop=True)
        if label:
            self.timings[label] = self.timings.get(label, 0.) + perf_counter() - tick
        return values.astype('float32')

    def add(self, values):
        values = values[[c for c in values.columns if c not in self.frame.columns]]
        self.frame = pd.concat([self.frame, values], axis=1)


def lgb_common(params, seed, bagging):
    extra = dict(subsample=.8, subsample_freq=1, colsample_bytree=.8) if bagging else {}
    return dict(learning_rate=params['learning_rate'], num_leaves=params['num_leaves'],
                min_child_samples=params['min_child_samples'], reg_lambda=params['reg_lambda'],
                n_estimators=params['n_estimators'], n_jobs=params['n_jobs'], random_state=seed,
                deterministic=True, force_col_wise=True, verbosity=-1, **extra)


def fit_ranker(train, val, columns, categorical, params, seed, bagging=False, trees=None):
    """LambdaRank on informative user-weeks; early stopping on validation full NDCG@10."""
    common = lgb_common(params, seed, bagging)
    if trees is not None:
        common['n_estimators'] = trees
    model = lgb.LGBMRanker(objective='lambdarank', metric='None', label_gain=[0, 1],
                           lambdarank_truncation_level=20, **common)
    rows = train.frame.iloc[train.rank_rows]
    kwargs = dict(group=train.rank_groups, categorical_feature=[c for c in columns if c in categorical])
    if val is not None:
        kwargs.update(eval_set=[(val.frame[columns], val.frame.label)], eval_group=[val.groups],
                      eval_metric=val.evaluator.lightgbm_metric,
                      callbacks=[lgb.early_stopping(params['early_stopping_rounds'], first_metric_only=True,
                                                    verbose=False)])
    tick = perf_counter()
    model.fit(rows[columns], rows.label, **kwargs)
    return model, perf_counter() - tick


def fit_binary(train, val, columns, categorical, params, seed, bagging=False):
    model = lgb.LGBMClassifier(objective='binary', **lgb_common(params, seed, bagging))
    tick = perf_counter()
    model.fit(train.frame[columns], train.frame.label, categorical_feature=[c for c in columns if c in categorical],
              eval_set=[(val.frame[columns], val.frame.label)], eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(params['early_stopping_rounds'], first_metric_only=True, verbose=False)])
    return model, perf_counter() - tick


def trees_of(model):
    return int(model.best_iteration_ or model.n_estimators)


def paired(a, b, repeats, seed):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(repeats, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return {'delta': float(d.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi),
            'p_left_better': float((means > 0).mean())}


def weighted_auc(inv, n_groups, y, w):
    """ROC-AUC with row weights; inv indexes ascending tie groups of the scores."""
    wp = np.bincount(inv, weights=w * y, minlength=n_groups)
    wn = np.bincount(inv, weights=w * (1 - y), minlength=n_groups)
    below = np.cumsum(wn) - wn
    return float((wp * (below + .5 * wn)).sum() / (wp.sum() * wn.sum()))


def row_logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_contrasts(evaluator, y, models, contrasts, repeats, seed):
    """Paired user bootstrap; one common resample of users for every model and metric.

    models[name] = (rank_scores, probabilities). NDCG/Recall are user means; ROC-AUC and
    LogLoss are pooled over the rows of the resampled users (each user's rows weighted
    by the user's multiplicity).
    """
    n = len(evaluator.users)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1 / n), size=repeats).astype(np.float64)
    ui = evaluator.user_index
    rows_per_user = np.bincount(ui, minlength=n).astype(float)
    needed = sorted({m for pair in contrasts for m in pair})
    samples, points = {}, {}
    for name in needed:
        scores, prob = models[name]
        per = evaluator.per_user(scores)
        loss = row_logloss(y, prob)
        loss_user = np.bincount(ui, weights=loss, minlength=n)
        _, inv = np.unique(scores, return_inverse=True)
        ng = inv.max() + 1
        points[name] = {'ndcg10': per.ndcg10.mean(), 'recall10': per.recall10.mean(),
                        'roc_auc': weighted_auc(inv, ng, y, np.ones(len(y))), 'logloss': loss.mean()}
        auc = np.array([weighted_auc(inv, ng, y, W[r][ui]) for r in range(repeats)])
        samples[name] = {'ndcg10': W @ per.ndcg10.to_numpy() / n, 'recall10': W @ per.recall10.to_numpy() / n,
                         'roc_auc': auc, 'logloss': (W @ loss_user) / (W @ rows_per_user)}
    rows = []
    for left, right in contrasts:
        for metric in METRICS:
            d = samples[left][metric] - samples[right][metric]
            lo, hi = np.quantile(d, [.025, .975])
            better = (d < 0) if metric == 'logloss' else (d > 0)
            rows.append({'left': left, 'right': right, 'metric': metric,
                         'left_value': float(points[left][metric]), 'right_value': float(points[right][metric]),
                         'delta': float(points[left][metric] - points[right][metric]),
                         'ci95_low': float(lo), 'ci95_high': float(hi), 'p_left_better': float(better.mean())})
    return pd.DataFrame(rows)


def metric_table(split_name, preds, evaluator, y, baselines):
    rows = []
    for name, (score, prob) in preds.items():
        per = evaluator.per_user(score)
        kind = 'baseline' if name in baselines else 'binary' if name.endswith('__binary') else 'lambdarank'
        rows.append({'split': split_name, 'model': name, 'kind': kind, 'ndcg10': float(per.ndcg10.mean()),
                     'recall10': float(per.recall10.mean()), 'roc_auc': float(roc_auc_score(y, score)),
                     'logloss': float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]))})
    return rows


def candidate_diagnostics(split):
    rows = []
    for name, frame, targets in [(split.name, split.frame, split.targets),
                                 *[(f'{split.name}_block{p.block}', p.frame, p.targets) for p in split.parts]]:
        ev = RankingEvaluator(frame, targets, 10)
        oracle = ev.per_user(frame.label.to_numpy(dtype=float))
        rows.append({'population': name, 'evaluation_users': len(ev.users), 'pairs': len(frame),
                     'target_pairs': int(ev.n_true.sum()), 'positive_pairs': int(frame.label.sum()),
                     'targets_in_pool_share': float(ev.candidate_hits.sum() / ev.n_true.sum()),
                     'candidate_recall_macro': float(oracle.candidate_recall.mean()),
                     'users_without_positive_share': float((ev.candidate_hits == 0).mean()),
                     'oracle_ndcg10': float(oracle.ndcg10.mean()), 'oracle_recall10': float(oracle.recall10.mean())})
    return rows


def feature_group(c, manual_groups):
    if c in COMMON_FEATURES:
        return 'candidate_scores'
    if c in manual_groups:
        return f'manual:{manual_groups[c]}'
    if c.startswith('v1_gen__'):
        return 'v1:' + c.split('__')[1]
    if c.startswith('gen__'):
        return 'v2_base:' + c.split('__')[1]
    if c.startswith('v2__'):
        return 'v2_extra:' + '_'.join(c.split('__')[1:3])
    if c.startswith('expr__'):
        return 'v3_formula'
    if c.startswith('g__'):
        return 'graph:' + ('svd' if 'svd' in c else 'recent_covis' if 'recent' in c else 'covis')
    if c.startswith('p__'):
        return 'graph:popularity'
    return 'other'


def run(output, size=20000, seed=2026, max_trees=1000, patience=50, repeats=2000, seeds=5, n_jobs=None,
        pipelines=PIPELINES):
    start = perf_counter()
    pipelines = [p for p in PIPELINES if p in set(pipelines)]
    if not pipelines:
        raise ValueError(f'choose pipelines from {PIPELINES}')
    run_v2 = 'v2' in pipelines or 'v4' in pipelines
    output = Path(output).resolve()
    if output.exists() and any(p.name != 'run.log' for p in output.iterdir()):
        raise FileExistsError(f'{output} is not empty; unified results are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir()
    (output / 'selection').mkdir()
    config = json.loads((ROOT / 'config.json').read_text())
    params = dict(config['model'], n_estimators=max_trees, early_stopping_rounds=patience,
                  n_jobs=n_jobs or os.cpu_count())
    search_seed = config['seed']
    write_json(output / 'protocol.json', {
        'created_utc': datetime.now(timezone.utc), 'permutation_seed': seed, 'block_size': size,
        'blocks': BLOCKS, 'unused_blocks': UNUSED_BLOCKS, 'train_cutoffs': TRAIN_CUTOFFS,
        'validation_cutoff': '2021-08-09', 'test_cutoff': '2021-08-16', 'target_days': 7,
        'model': params, 'final_bagging_seeds': [1000 + s for s in range(seeds)],
        'final_bagging': {'subsample': .8, 'subsample_freq': 1, 'colsample_bytree': .8},
        'search_model_seed': search_seed, 'search_bagging': False,
        'generator_fit': 'train-user history before 2021-07-12',
        'v1': {'windows': V1_WINDOWS, 'settings': V1_SETTINGS},
        'v2_search': {'windows': V2_WINDOWS, 'variants': VARIANTS,
                      'criterion': 'validation NDCG@10 of the +scores ranker; ties: fewer features'},
        'v3_search': {'seeds': 128, 'mutations': 64, 'enrich': 64, 'attribute_lift': 'all relations',
                      'probe_trees': V3_PROBE_TREES, 'subsets': [*V3_PREFIXES, 'full_pool'],
                      'ranking': 'train-only gain of an 80-tree probe on the pruned pool',
                      'criterion': 'validation NDCG@10 of the +scores ranker; ties: fewer features'},
        'v4_search': {'base': 'selected V2 +scores', 'families': ['covis', 'recent_covis', 'svd', 'popularity'],
                      'rule': f'greedy by single-family validation gain; keep if delta>0 and bootstrap P>{V4_THRESHOLD}'},
        'ablations': {'base': 'manual+scores', 'groups': {**PRIMARY, **INDIRECT},
                      'note': 'drop-one-group on validation; reported only, manual set unchanged'},
        'pipelines': pipelines, 'contrasts': CONTRASTS, 'metrics': METRICS, 'bootstrap_repeats': repeats})

    excluded = pd.read_csv(ROOT / config['excluded_users_path']).user_id.to_numpy()
    raw, users, items = load_raw(config)
    dates = split_dates(raw)
    val_cutoff, val_end = dates['validation']
    test_cutoff, test_end = dates['test']
    assert str(val_cutoff.date()) == '2021-08-09' and str(test_cutoff.date()) == '2021-08-16'
    blocks, pops = assign_blocks(raw, dates['train'][0], excluded, size, seed)
    for split, ids in BLOCKS.items():
        pd.DataFrame({'user_id': np.concatenate(pops[split]),
                      'block': np.repeat(ids, size)}).to_csv(output / f'{split}_users.csv', index=False)
    test_users = pops.pop('test')  # loaded only after plan.json

    train_users = pops['train'][0]
    train_events = raw.loc[raw.user_id.isin(train_users) & (raw.last_watch_dt < val_cutoff)].copy()
    fit_cutoff = pd.Timestamp(TRAIN_CUTOFFS[0])
    train = Split('train', [Part(train_events, users, items, pd.Timestamp(c), pd.Timestamp(c) + pd.Timedelta(days=7),
                                 train_users, config, 'train', 1) for c in TRAIN_CUTOFFS])
    val_parts = []
    for b, population in zip(BLOCKS['validation'], pops['validation']):
        ev = raw.loc[raw.user_id.isin(population) & (raw.last_watch_dt < val_end)].copy()
        val_parts.append(Part(ev, users, items, val_cutoff, val_end, population, config, 'validation', b))
    val = Split('validation', val_parts)
    splits = {'train': train, 'validation': val}
    categorical = set(CATEGORICAL_FEATURES)
    search_time = {}
    vu = val.evaluator.users

    def search_fit(columns):
        model, fit_s = fit_ranker(train, val, columns, categorical, params, search_seed)
        per = val.evaluator.per_user(model.predict(val.frame[columns])).ndcg10.to_numpy()
        return per, {'features': len(columns), 'trees': trees_of(model), 'val_ndcg10': float(per.mean()),
                     'fit_s': fit_s}

    v1_cols, v2_cols, v3_cols, v4_extra, chosen, selected = [], [], [], [], [], {}
    # ---- manual reference and ablations (validation only) ----
    if 'manual' in pipelines:
        tick = perf_counter()
        manual_scores = [*COMMON_FEATURES, *MANUAL_FEATURES]
        base_per, base_info = search_fit(manual_scores)
        abl = [{'group': 'none', 'scope': 'full', 'dropped': 0, **base_info}]
        for scope, groups in [('primary', PRIMARY), ('indirect', INDIRECT)]:
            for g, cols in groups.items():
                per, info = search_fit([c for c in manual_scores if c not in cols])
                abl.append({'group': g, 'scope': scope, 'dropped': len(cols), **info,
                            **{f'{k}_vs_full': v for k, v in paired(per, base_per, repeats, seed).items()}})
                log(f"ablation -{g}: {info['val_ndcg10']:.4f} ({info['val_ndcg10'] - base_info['val_ndcg10']:+.4f})")
        pd.DataFrame(abl).to_csv(output / 'ablations.csv', index=False)
        search_time['ablations'] = perf_counter() - tick

    # ---- V1: schema features with fixed windows ----
    if 'v1' in pipelines:
        tick = perf_counter()
        v1 = FeatureGenerator(windows=V1_WINDOWS, **V1_SETTINGS).fit(train_events, fit_cutoff, users, items)
        rename = {c: f'v1_{c}' for c in v1.columns}
        for s in splits.values():
            s.add(s.compute(lambda p: v1.transform(p.events, p.cutoff, p.keys, users, items)[v1.columns]
                            .rename(columns=rename), 'v1'))
        v1_cols, v1_dropped = prune_train(train.frame, list(rename.values()), [rename[c] for c in v1.categorical])
        categorical |= {rename[c] for c in v1.categorical}
        write_json(output / 'selection/v1_schema.json', v1.schema)
        write_json(output / 'selection/v1_pruning.json', {'features': v1_cols, 'dropped': v1_dropped})
        search_time['v1_fit_prune'] = perf_counter() - tick - train.timings['v1'] - val.timings['v1']

    # ---- V2: windows x families searched on validation ----
    if run_v2:
        tick = perf_counter()
        trials, best = [], None
        for windows in V2_WINDOWS:
            for variant, fams in VARIANTS.items():
                gen = FeatureGeneratorV2(families=fams, windows=windows).fit(train_events, fit_cutoff, users, items)
                vals, transform_s = {}, {}
                for k, s in splits.items():
                    t0 = perf_counter()
                    vals[k] = s.compute(lambda p: gen.transform(p.events, p.cutoff, p.keys, users, items)[gen.columns])
                    transform_s[k] = perf_counter() - t0
                cols, _ = prune_train(vals['train'], gen.columns, gen.categorical)
                for k, s in splits.items():
                    s.frame = pd.concat([s.frame, vals[k][[c for c in cols if c not in s.frame.columns]]], axis=1)
                per, info = search_fit([*COMMON_FEATURES, *cols])
                row = {'windows': list(windows), 'variant': variant, 'recipes': len(gen.columns), **info,
                       **{f'transform_{k}_s': v for k, v in transform_s.items()}}
                trials.append(row)
                log(f"V2 {windows} {variant}: {len(cols)} feats, {info['trees']} trees, val NDCG {info['val_ndcg10']:.4f}")
                key = (-info['val_ndcg10'], len(cols), tuple(windows), variant)
                keep_cols = set(MANUAL_FEATURES) | set(COMMON_FEATURES) | set(v1_cols)
                if best is None or key < best['key']:
                    best = {'key': key, 'gen': gen, 'cols': cols, 'per': per, 'row': row, 'transform_s': transform_s}
                keep_cols |= set(best['cols'])
                for s in splits.values():  # drop non-winning V2 columns to bound memory
                    drop = [c for c in s.frame.columns if (c.startswith('gen__') or c.startswith('v2__')) and c not in keep_cols]
                    s.frame = s.frame.drop(columns=drop)
        pd.DataFrame(trials).to_csv(output / 'selection/v2_search.csv', index=False)
        v2, v2_cols, v2_per = best['gen'], best['cols'], best['per']
        categorical |= set(v2.categorical)
        for k, s in splits.items():
            s.timings['v2'] = best['transform_s'][k]
        write_json(output / 'selection/v2_schema.json', v2.schema)
        search_time['v2_search'] = perf_counter() - tick - sum(best['transform_s'].values())
        log(f"V2 selected: {best['row']['windows']} {best['row']['variant']}, {len(v2_cols)} features")

    # ---- V3: formula pool fit on train history, subset chosen on validation ----
    if 'v3' in pipelines:
        tick = perf_counter()
        v3 = FormulaGenerator().fit(train_events, fit_cutoff, users, items)
        n_seed = len(v3.columns)
        seed_vals = train.compute(lambda p: v3.transform(p.events, p.cutoff, p.keys, users, items)[v3.columns])
        seed_cols, _ = prune_train(seed_vals, v3.columns, v3.categorical)
        probe_frame = pd.concat([train.frame[['user_id', 'item_id', 'week', 'label']], seed_vals], axis=1)
        probe_split = Split.__new__(Split)
        probe_split.frame, probe_split.rank_rows, probe_split.rank_groups = probe_frame, train.rank_rows, train.rank_groups
        probe, _ = fit_ranker(probe_split, None, seed_cols, set(v3.categorical), params, search_seed, trees=V3_PROBE_TREES)
        gains = dict(zip(seed_cols, probe.booster_.feature_importance('gain')))
        mutations = v3.mutate(sorted(seed_cols, key=lambda c: (-gains[c], c)))
        extra = v3.enrich()
        n_before_lift = len(v3.columns)
        for field in v3.profiler.schema['relations']:
            v3.add(attribute_lift(field, window(None)))
        del seed_vals, probe_frame, probe_split
        for s in splits.values():
            s.add(s.compute(lambda p: v3.transform(p.events, p.cutoff, p.keys, users, items)[v3.columns], 'v3_pool'))
        pool_cols, pool_dropped = prune_train(train.frame, v3.columns, v3.categorical)
        categorical |= set(v3.categorical)
        probe, _ = fit_ranker(train, None, pool_cols, categorical, params, search_seed, trees=V3_PROBE_TREES)
        gains = dict(zip(pool_cols, probe.booster_.feature_importance('gain')))
        ranked = sorted(pool_cols, key=lambda c: (-gains[c], c))
        pd.DataFrame({'feature': ranked, 'train_gain': [gains[c] for c in ranked]}).to_csv(
            output / 'selection/v3_pool_gain.csv', index=False)
        v3_trials, best3 = [], None
        for label, cols in [*[(f'top{n}', ranked[:n]) for n in V3_PREFIXES if n < len(ranked)], ('full_pool', ranked)]:
            per, info = search_fit([*COMMON_FEATURES, *cols])
            v3_trials.append({'subset': label, **info})
            log(f"V3 {label}: {len(cols)} feats, {info['trees']} trees, val NDCG {info['val_ndcg10']:.4f}")
            key = (-info['val_ndcg10'], len(cols))
            if best3 is None or key < best3['key']:
                best3 = {'key': key, 'label': label, 'cols': cols, 'per': per}
        pd.DataFrame(v3_trials).to_csv(output / 'selection/v3_search.csv', index=False)
        v3_cols = best3['cols']
        v3_schema = v3.export()
        write_json(output / 'selection/v3_pool_schema.json', v3_schema)
        v3_schema['expressions'] = [r for r in v3_schema['expressions'] if r['name'] in v3_cols]
        write_json(output / 'selection/v3_schema.json', v3_schema)
        v3_pool = {'seed_expressions': n_seed, 'seed_after_pruning': len(seed_cols), 'mutations': len(mutations),
                   'enriched': len(extra), 'attribute_lifts_added': len(v3.columns) - n_before_lift,
                   'pool_expressions': len(v3.columns), 'pool_after_pruning': len(pool_cols),
                   'pruned': pool_dropped, 'selected_subset': best3['label'], 'selected_features': len(v3_cols)}
        write_json(output / 'selection/v3_pool.json', v3_pool)
        for s in splits.values():
            s.frame = s.frame.drop(columns=[c for c in v3.columns if c not in v3_cols and c in s.frame.columns])
        search_time['v3_search'] = perf_counter() - tick - train.timings['v3_pool'] - val.timings['v3_pool']
        log(f"V3 selected: {best3['label']}, {len(v3_cols)} features")

    # ---- V4: V2 + graph families, greedy on validation ----
    if 'v4' in pipelines:
        tick = perf_counter()
        graph, pop = GraphFamily(), FilteredPopularity()
        for s in splits.values():
            s.add(s.compute(lambda p: graph.transform(p.events, p.cutoff, p.keys), 'graph'))
            s.add(s.compute(lambda p: pop.transform(p.events, p.cutoff, p.keys), 'popularity'))
        fams = graph_families(graph, pop)
        v2_scores = [*COMMON_FEATURES, *v2_cols]
        sel_log, single = [], {}
        for name, cols in fams.items():
            per, info = search_fit([*v2_scores, *cols])
            single[name] = paired(per, v2_per, repeats, seed)
            sel_log.append({'step': 'single', 'family': name, **info, **single[name]})
            log(f"V4 single {name}: {single[name]['delta']:+.4f} P={single[name]['p_left_better']:.2f}")
        chosen, current, current_per = [], list(v2_scores), v2_per
        for name in sorted(fams, key=lambda n: -single[n]['delta']):
            cols = [*current, *fams[name]]
            per, info = search_fit(cols)
            r = paired(per, current_per, repeats, seed)
            keep = r['delta'] > 0 and r['p_left_better'] > V4_THRESHOLD
            sel_log.append({'step': 'greedy', 'family': name, 'kept': keep, **info, **r})
            log(f"V4 greedy +{name}: {r['delta']:+.4f} P={r['p_left_better']:.2f} -> {'keep' if keep else 'drop'}")
            if keep:
                chosen.append(name)
                current, current_per = cols, per
        pd.DataFrame(sel_log).to_csv(output / 'selection/v4_selection.csv', index=False)
        v4_extra = [c for f in chosen for c in fams[f]]
        search_time['v4_search'] = perf_counter() - tick - sum(s.timings.get(k, 0) for s in splits.values()
                                                               for k in ['graph', 'popularity'])

    base_sets = {}
    if 'manual' in pipelines:
        base_sets.update({'manual_core': list(CORE_FEATURES), 'manual': list(MANUAL_FEATURES),
                          'manual_noib': [c for c in MANUAL_FEATURES if c not in PRIMARY['item_behavior']]})
        selected['manual'] = {k: len(v) for k, v in base_sets.items()}
    if 'v1' in pipelines:
        base_sets['auto_v1'] = v1_cols
        selected['v1'] = {'windows': V1_WINDOWS, 'features': len(v1_cols)}
    if run_v2:
        selected['v2'] = {'windows': best['row']['windows'], 'variant': best['row']['variant'],
                          'features': len(v2_cols), 'val_ndcg10_search': best['row']['val_ndcg10']}
    if 'v2' in pipelines:
        base_sets['auto_v2'] = v2_cols
    if 'v3' in pipelines:
        base_sets['auto_v3'] = v3_cols
        selected['v3'] = v3_pool | {'val_ndcg10_search': float(best3['per'].mean())}
    if 'v4' in pipelines:
        base_sets['auto_v4'] = [*v2_cols, *v4_extra]
        selected['v4'] = {'families': chosen, 'extra_features': len(v4_extra)}
    if 'manual' in pipelines and 'v2' in pipelines:
        base_sets['manual_auto_v2'] = [*MANUAL_FEATURES, *v2_cols]
    sets = dict(base_sets)
    sets.update({f'{k}+scores': [*COMMON_FEATURES, *v] for k, v in base_sets.items()})
    contrasts_run = [(l, r) for l, r in CONTRASTS if all(m == 'popular' or m in sets for m in (l, r))]
    write_json(output / 'feature_sets.json', sets)
    write_json(output / 'selection/selected.json', selected)

    # ---- final models: 5 bagging seeds, LambdaRank + binary, all on the same split ----
    models, records, val_pred, gains_rows = {}, [], {}, []
    yv = val.frame.label.to_numpy()
    fitted = {}
    for name, cols in sets.items():
        same = fitted.get(tuple(cols))
        if same is not None:  # e.g. V4 with no family kept is exactly V2: reuse, do not refit
            models[name] = models[same]
            val_pred[name], val_pred[f'{name}__binary'] = val_pred[same], val_pred[f'{same}__binary']
            records.append({**next(r for r in records if r['model'] == same), 'model': name, 'alias_of': same})
            gains_rows.append(pd.concat([g for g in gains_rows if g['set'].iat[0] == same]).assign(set=name))
            log(f'final {name}: identical columns to {same}, models reused')
            continue
        fitted[tuple(cols)] = name
        rankers, binaries, rs, ps = [], [], [], []
        fit_s = 0.
        for s in range(seeds):
            r, t1 = fit_ranker(train, val, cols, categorical, params, 1000 + s, bagging=True)
            b, t2 = fit_binary(train, val, cols, categorical, params, 1000 + s, bagging=True)
            fit_s += t1 + t2
            r.booster_.save_model(str(output / 'models' / f'{name}__ranker__seed{s}.txt'), num_iteration=trees_of(r))
            b.booster_.save_model(str(output / 'models' / f'{name}__binary__seed{s}.txt'), num_iteration=trees_of(b))
            rankers.append(r)
            binaries.append(b)
            rs.append(r.predict(val.frame[cols]))
            ps.append(b.predict_proba(val.frame[cols])[:, 1])
            gains_rows.append(pd.DataFrame({'set': name, 'seed': s, 'feature': cols,
                                            'gain': r.booster_.feature_importance('gain', iteration=trees_of(r))}))
        score, prob = np.mean(rs, axis=0), np.mean(ps, axis=0)
        platt = LogisticRegression(C=1e6).fit(score.reshape(-1, 1), yv)
        models[name] = (rankers, binaries, platt, cols)
        val_pred[name] = (score, platt.predict_proba(score.reshape(-1, 1))[:, 1])
        val_pred[f'{name}__binary'] = (prob, prob)
        records.append({'model': name, 'features': len(cols), 'fit_s': fit_s,
                        'ranker_trees': [trees_of(r) for r in rankers], 'binary_trees': [trees_of(b) for b in binaries],
                        'val_ndcg10': val.evaluator.summary(score)['ndcg10']})
        log(f"final {name}: {len(cols)} feats, ranker trees {records[-1]['ranker_trees']}, "
            f"val NDCG ({seeds}-seed mean score) {records[-1]['val_ndcg10']:.4f}")
    pd.DataFrame(records).to_csv(output / 'training.csv', index=False)

    def knn_score(frame):
        return -frame.knn_rank.to_numpy(dtype=float) * (frame.knn_rank.max() + 1) - frame.popular_rank.to_numpy(dtype=float)
    base_platt = {}
    for name, fn in [('popular', lambda f: -f.popular_rank.to_numpy(dtype=float)), ('itemknn', knn_score)]:
        x = fn(val.frame)
        base_platt[name] = (fn, LogisticRegression(C=1e6).fit(x.reshape(-1, 1), yv))
        val_pred[name] = (x, base_platt[name][1].predict_proba(x.reshape(-1, 1))[:, 1])
    pd.DataFrame({'user_id': val.frame.user_id, 'item_id': val.frame.item_id, 'label': yv,
                  **{k: v[0] for k, v in val_pred.items()}}).to_parquet(output / 'validation_predictions.parquet', index=False)

    metric_rows = metric_table('validation', val_pred, val.evaluator, yv, base_platt)
    diag = candidate_diagnostics(val)
    comp = {'train': train.timings, 'validation': val.timings}
    plan = {'created_utc': datetime.now(timezone.utc), 'status': 'frozen_before_test',
            'feature_sets': {k: len(v) for k, v in sets.items()}, 'feature_sets_file': 'feature_sets.json',
            'schemas': {'v1': 'selection/v1_schema.json', 'v2': 'selection/v2_schema.json',
                        'v3': 'selection/v3_schema.json', 'v4_families': chosen},
            'selected': selected, 'contrasts': contrasts_run, 'primary_contrast': CONTRASTS[0],
            'metrics': METRICS, 'kinds': ['lambdarank', 'binary'], 'bootstrap_repeats': repeats,
            'test_blocks': BLOCKS['test'], 'test_cutoff': str(test_cutoff.date()), 'seeds': seeds,
            'test_opened': False}
    write_json(output / 'plan.json', plan)
    write_json(output / 'status.json', {'status': 'models_saved_test_unopened'})
    del train, val, splits

    # ---- test opened once; nothing below changes a model or a feature set ----
    log('Opening test users (blocks 6+7)')
    test_parts = []
    for b, population in zip(BLOCKS['test'], test_users):
        ev = raw.loc[raw.user_id.isin(population) & (raw.last_watch_dt < test_end)].copy()
        test_parts.append(Part(ev, users, items, test_cutoff, test_end, population, config, 'test', b))
    del raw
    test = Split('test', test_parts)
    if 'v1' in pipelines:
        test.add(test.compute(lambda p: v1.transform(p.events, p.cutoff, p.keys, users, items)[v1.columns]
                              .rename(columns=rename)[v1_cols], 'v1'))
    if run_v2:
        test.add(test.compute(lambda p: v2.transform(p.events, p.cutoff, p.keys, users, items)[v2_cols], 'v2'))
    if 'v3' in pipelines:
        test.add(test.compute(lambda p: v3.transform(p.events, p.cutoff, p.keys, users, items, v3_cols)[v3_cols], 'v3'))
    if any(f != 'popularity' for f in chosen):
        test.add(test.compute(lambda p: graph.transform(p.events, p.cutoff, p.keys), 'graph'))
    if 'popularity' in chosen:
        test.add(test.compute(lambda p: pop.transform(p.events, p.cutoff, p.keys), 'popularity'))
    yt = test.frame.label.to_numpy()
    test_pred, inference = {}, {}
    for name, (rankers, binaries, platt, cols) in models.items():
        tick = perf_counter()
        score = np.mean([r.predict(test.frame[cols]) for r in rankers], axis=0)
        t_r = perf_counter() - tick
        tick = perf_counter()
        prob = np.mean([b.predict_proba(test.frame[cols])[:, 1] for b in binaries], axis=0)
        inference[name] = {'ranker_s': t_r, 'binary_s': perf_counter() - tick}
        test_pred[name] = (score, platt.predict_proba(score.reshape(-1, 1))[:, 1])
        test_pred[f'{name}__binary'] = (prob, prob)
    for name, (fn, platt) in base_platt.items():
        x = fn(test.frame)
        test_pred[name] = (x, platt.predict_proba(x.reshape(-1, 1))[:, 1])
    pd.DataFrame({'user_id': test.frame.user_id, 'item_id': test.frame.item_id, 'label': yt,
                  **{k: v[0] for k, v in test_pred.items()}}).to_parquet(output / 'test_predictions.parquet', index=False)

    # ---- metrics, contrasts, diagnostics ----
    metrics = pd.DataFrame(metric_rows + metric_table('test', test_pred, test.evaluator, yt, base_platt))
    metrics.to_csv(output / 'metrics.csv', index=False)
    binary_contrasts = [(l if l == 'popular' else f'{l}__binary', r if r == 'popular' else f'{r}__binary')
                        for l, r in contrasts_run]
    contrasts = pd.concat([
        bootstrap_contrasts(test.evaluator, yt, test_pred, contrasts_run, repeats, seed).assign(kind='lambdarank'),
        bootstrap_contrasts(test.evaluator, yt, test_pred, binary_contrasts, repeats, seed).assign(kind='binary')],
        ignore_index=True)
    contrasts.to_csv(output / 'contrasts.csv', index=False)
    pd.DataFrame(diag + candidate_diagnostics(test)).to_csv(output / 'candidate_diagnostics.csv', index=False)

    # ---- importance by feature and by group ----
    manual_groups = {c: g for g, cols in {**PRIMARY, **INDIRECT}.items() for c in cols if c not in COMMON_FEATURES}
    gains_df = pd.concat(gains_rows, ignore_index=True)
    top = gains_df.groupby(['set', 'feature'], sort=False).gain.mean().reset_index()
    top['share'] = top.gain / top.groupby('set').gain.transform('sum')
    top['group'] = top.feature.map(lambda c: feature_group(c, manual_groups))
    top = top.sort_values(['set', 'gain'], ascending=[True, False])
    top.to_csv(output / 'top_features.csv', index=False)
    top.groupby(['set', 'group']).share.sum().reset_index().sort_values(['set', 'share'], ascending=[True, False]).to_csv(
        output / 'gain_groups.csv', index=False)

    # ---- timings ----
    comp['test'] = test.timings
    uses = {'manual_core': ['manual'], 'manual': ['manual'], 'manual_noib': ['manual'], 'auto_v1': ['v1'], 'auto_v2': ['v2'],
            'auto_v3': ['v3_pool', 'v3'], 'auto_v4': ['v2', 'graph', 'popularity'], 'manual_auto_v2': ['manual', 'v2']}
    search_of = {'auto_v1': 'v1_fit_prune', 'auto_v2': 'v2_search', 'auto_v3': 'v3_search', 'auto_v4': 'v4_search'}
    rec = {r['model']: r for r in records}
    per_set = {}
    for name in sets:
        base = name.replace('+scores', '')
        per_set[name] = {
            'features': len(sets[name]),
            'feature_build_s': {sp: sum(t.get(k, 0.) for k in uses[base]) for sp, t in comp.items()},
            'search_s': search_time.get(search_of.get(base, ''), 0.) + (search_time['v2_search'] if base == 'auto_v4' else 0.),
            'fit_s_all_seeds': rec[name]['fit_s'], 'test_inference_s': inference[name]}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if platform.system() == 'Darwin' else 1024)
    write_json(output / 'timings.json', {'components': comp, 'search': search_time, 'sets': per_set,
                                         'elapsed_s': perf_counter() - start, 'peak_rss_mb': peak})
    summary = {'status': 'complete', 'elapsed_s': perf_counter() - start, 'peak_rss_mb': peak,
               'block_size': size, 'validation_users': len(vu), 'test_users': len(test.evaluator.users),
               'test_pairs': len(test.frame), 'selected': selected,
               'python': platform.python_version(), 'lightgbm': lgb.__version__, 'pandas': pd.__version__,
               'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
               'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'n_jobs': params['n_jobs'],
               'finished_utc': datetime.now(timezone.utc)}
    write_json(output / 'summary.json', summary)
    write_json(output / 'status.json', {'status': 'complete', 'test_opened_once': True})
    log('\n' + metrics.loc[metrics.split == 'test'].round(4).to_string(index=False))
    if len(contrasts):
        log('\n' + contrasts.loc[contrasts.metric == 'ndcg10'].round(4).to_string(index=False))
    return summary
