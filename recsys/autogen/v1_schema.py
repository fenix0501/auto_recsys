"""Small schema-driven compiler for aggregated user-item sources, without manual FE.

The caller declares only table/key/time roles. Types, category relations and
feature recipes are inferred on the training history and frozen thereafter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class FeatureBudgetWarning(UserWarning):
    """The explicit feature budget omitted otherwise valid recipes."""


@dataclass(frozen=True)
class Roles:
    user: str = 'user_id'
    item: str = 'item_id'
    time: str = 'last_watch_dt'


@dataclass(frozen=True)
class FeatureSpec:
    relation: str
    aggregation: str
    field: str | None = None
    window_days: int | None = None
    reduction: str | None = None

    @property
    def name(self):
        # Include a digest so sanitizing arbitrary source names cannot collide.
        body = asdict(self)
        suffix = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:8]
        parts = [self.relation, self.field or 'rows', self.aggregation,
                 f'{self.window_days}d' if self.window_days else 'all', self.reduction or 'value']
        return 'gen__' + '__'.join(re.sub(r'[^a-zA-Z0-9_]', '_', p) for p in parts) + '__' + suffix


def tokens(value):
    if pd.isna(value):
        return []
    return sorted({s.strip() for s in str(value).split(',') if s.strip()})


def prune_train(frame, columns, categorical=()):
    """Train-only exact pruning; codes and continuous values are distinct types."""
    kept, dropped, buckets = [], [], {}
    for col in columns:
        if frame[col].nunique(dropna=False) <= 1:
            dropped.append({'feature': col, 'reason': 'constant_or_all_missing'})
            continue
        key = (col in categorical, hashlib.sha256(pd.util.hash_pandas_object(
            frame[col].astype(float), index=False).to_numpy().tobytes()).hexdigest())
        duplicate = next((k for k in buckets.get(key, [])
                          if frame[col].astype(float).equals(frame[k].astype(float))), None)
        if duplicate:
            dropped.append({'feature': col, 'reason': 'exact_duplicate', 'duplicate_of': duplicate})
        else:
            kept.append(col)
            buckets.setdefault(key, []).append(col)
    return kept, dropped


class FeatureGenerator:
    def __init__(self, roles=Roles(), windows=(7, 30), max_features=64,
                 max_relations=5, max_event_numeric=2, max_metadata_numeric=4,
                 max_categories=32, max_tokens=4, max_vocab=8192):
        if not windows or any(not isinstance(w, int) or w <= 0 for w in windows):
            raise ValueError('Windows must be positive integer days')
        if max_features < 1:
            raise ValueError('Feature budget must be positive')
        self.roles = roles
        self.windows = tuple(sorted(set(windows)))
        self.max_features = max_features
        self.max_relations = max_relations
        self.max_event_numeric = max_event_numeric
        self.max_metadata_numeric = max_metadata_numeric
        self.max_categories = max_categories
        self.max_tokens = max_tokens
        self.max_vocab = max_vocab

    def history(self, events, cutoff):
        r = self.roles
        history = events.loc[events[r.time] < cutoff].copy()
        if history[[r.user, r.item, r.time]].isna().any().any():
            raise ValueError('Historical keys/time cannot be missing')
        if history.duplicated([r.user, r.item]).any():
            raise ValueError('This source contract expects one aggregated user-item row')
        return history

    def _metadata(self, frame, key, ids):
        # Ignore future-only metadata rows even during schema inference.
        frame = frame.loc[frame[key].isin(ids)].copy()
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError('Metadata keys must be unique and nonmissing')
        return frame

    def fit(self, events, cutoff, users, items):
        r = self.roles
        history = self.history(events, cutoff)
        if history.empty:
            raise ValueError('Schema inference needs nonempty training history')
        meta = {'user': self._metadata(users, r.user, history[r.user]),
                'item': self._metadata(items, r.item, history[r.item])}
        self.schema = {'roles': asdict(r), 'fit_cutoff': str(cutoff), 'windows': self.windows,
                       'inference': [], 'event_numeric': [], 'numeric': {}, 'categories': {},
                       'relations': []}
        self.encodings = {}
        keys = {r.user, r.item, r.time}
        for col in sorted(set(history.columns) - keys):
            s = history[col]
            if is_numeric_dtype(s) and s.nunique() > 1:
                if len(self.schema['event_numeric']) < self.max_event_numeric:
                    self.schema['event_numeric'].append(col)
                else:
                    self.schema['inference'].append({'table': 'events', 'field': col, 'skip': 'numeric_budget'})
        relations = []
        for entity, frame in meta.items():
            self.schema['numeric'][entity] = []
            self.schema['categories'][entity] = []
            for col in sorted(set(frame.columns) - keys):
                s = frame[col].dropna()
                record = {'table': entity, 'field': col, 'nonmissing': len(s), 'unique': s.nunique()}
                if s.nunique() <= 1:
                    record['skip'] = 'constant_or_empty'
                elif is_numeric_dtype(s):
                    if len(self.schema['numeric'][entity]) < self.max_metadata_numeric:
                        self.schema['numeric'][entity].append(col)
                        record['kind'] = 'numeric'
                    else:
                        record['skip'] = 'numeric_budget'
                else:
                    s = s.astype('string')
                    lists = s.map(tokens)
                    vocab = {v for values in lists for v in values}
                    # Generic short categorical/list fields; exclude prose and
                    # very long token bags without any field-name allowlist.
                    is_list = s.str.contains(',', regex=False).mean() >= .05
                    eligible = (bool(vocab) and len(vocab) <= self.max_vocab
                                and lists.map(len).median() <= self.max_tokens
                                and np.median([len(v) for v in vocab]) <= 48
                                and (is_list or s.nunique() <= self.max_categories))
                    record.update(list_like=bool(is_list), token_vocabulary=len(vocab),
                                  median_tokens=float(lists.map(len).median()))
                    if not is_list and s.nunique() <= self.max_categories:
                        self.schema['categories'][entity].append(col)
                        self.encodings[f'{entity}.{col}'] = {v: i for i, v in enumerate(sorted(s.unique()))}
                        record['kind'] = 'categorical'
                    if entity == 'item' and eligible:
                        relations.append({'field': col, 'coverage': len(s) / len(frame),
                                          'vocabulary': len(vocab)})
                        record['relation_eligible'] = True
                    if 'kind' not in record and not record.get('relation_eligible'):
                        record['skip'] = 'text_or_high_cardinality'
                self.schema['inference'].append(record)
        # Prefer coverage, then compact relations, never labels or feature gains.
        relations.sort(key=lambda d: (-d['coverage'], d['vocabulary'], d['field']))
        self.schema['relations'] = [d['field'] for d in relations[:self.max_relations]]
        specs = []
        for entity in ['user', 'item']:
            for col in self.schema['numeric'][entity]:
                specs.append(FeatureSpec(entity, 'raw_numeric', col))
            for col in self.schema['categories'][entity]:
                specs.append(FeatureSpec(entity, 'raw_category', col))
            for window in [None, *self.windows]:
                specs.append(FeatureSpec(entity, 'count', window_days=window))
                specs.extend(FeatureSpec(entity, 'mean', col, window) for col in self.schema['event_numeric'])
            specs.append(FeatureSpec(entity, 'recency'))
        for window in [None, *self.windows]:
            specs.append(FeatureSpec('pair', 'count', window_days=window))
        specs.extend(FeatureSpec('numeric_taste', 'absolute_gap', col)
                     for col in self.schema['numeric']['item'])
        # Round-robin allocation avoids dropping entire later relations at cap.
        for window in self.windows:
            for reduction in ['mean', 'max']:
                specs.extend(FeatureSpec('user_attribute', 'share', col, window, reduction)
                             for col in self.schema['relations'])
        specs.extend(FeatureSpec('user_attribute', 'recency', col, reduction='min')
                     for col in self.schema['relations'])
        self.specs = specs[:self.max_features]
        omitted = specs[self.max_features:]
        if omitted:
            warnings.warn(
                f'Feature budget {self.max_features} keeps {len(self.specs)} of {len(specs)} recipes; '
                f'{len(omitted)} omitted recipes are recorded in schema.truncated_recipes',
                FeatureBudgetWarning, stacklevel=2)
        self.schema.update(feature_budget=self.max_features, potential_recipes=len(specs),
                           truncated_recipes=[asdict(s) | {'name': s.name} for s in omitted],
                           recipes=[asdict(s) | {'name': s.name} for s in self.specs],
                           encodings=self.encodings,
                           generator_settings={k: getattr(self, k) for k in [
                               'windows', 'max_features', 'max_relations', 'max_event_numeric',
                               'max_metadata_numeric', 'max_categories', 'max_tokens', 'max_vocab']})
        self.columns = [s.name for s in self.specs]
        self.categorical = [s.name for s in self.specs if s.aggregation == 'raw_category']
        return self

    @classmethod
    def from_schema(cls, schema):
        """Restore exported recipes and training vocabularies without refitting."""
        generator = cls(Roles(**schema['roles']), **schema['generator_settings'])
        generator.schema = schema
        generator.encodings = schema['encodings']
        generator.specs = [FeatureSpec(**{k: v for k, v in s.items() if k != 'name'})
                           for s in schema['recipes']]
        if [s.name for s in generator.specs] != [s['name'] for s in schema['recipes']]:
            raise ValueError('Exported feature recipe names do not match')
        generator.columns = [s.name for s in generator.specs]
        generator.categorical = [s.name for s in generator.specs if s.aggregation == 'raw_category']
        return generator

    def transform(self, events, cutoff, candidates, users, items):
        if not hasattr(self, 'specs'):
            raise ValueError('Fit and freeze the schema on training history first')
        if pd.Timestamp(cutoff) < pd.Timestamp(self.schema['fit_cutoff']):
            raise ValueError('Cannot transform earlier than schema fit cutoff')
        r = self.roles
        history = self.history(events, cutoff)
        out = candidates[[r.user, r.item]].copy().reset_index(drop=True)
        if out.duplicated([r.user, r.item]).any():
            raise ValueError('Duplicate candidate pairs')
        for key in [r.user, r.item]:
            if not set(out[key]).issubset(set(history[key])):
                raise ValueError('Candidates must have historical users and items')
        meta = {'user': self._metadata(users, r.user, history[r.user]).set_index(r.user),
                'item': self._metadata(items, r.item, history[r.item]).set_index(r.item)}
        windows = {None: history}
        windows.update({w: history[history[r.time] >= cutoff - pd.Timedelta(days=w)] for w in self.windows})
        cache = {}
        def aggregate(entity, window, agg, field=None):
            key = (entity, window, agg, field)
            if key not in cache:
                cols = [r.user, r.item] if entity == 'pair' else [getattr(r, entity)]
                g = windows[window].groupby(cols, sort=False)
                cache[key] = g.size() if agg == 'count' else g[field].agg(agg)
            return cache[key]
        # Distinct item-token edges. A source event counts once per token;
        # denominators always count original events, before any explosion.
        edges, queries, known = {}, {}, {}
        for field in self.schema['relations']:
            lists = meta['item'][field].map(tokens)
            edges[field] = lists.rename('_token').reset_index().explode('_token').dropna(subset=['_token'])
            known[field] = out[r.item].map(lists.map(len)).fillna(0).gt(0).to_numpy()
            queries[field] = out.assign(_row=np.arange(len(out))).merge(edges[field], on=r.item, how='left')
        relations = {}
        for spec in self.specs:
            if spec.aggregation.startswith('raw_'):
                key = getattr(r, spec.relation)
                values = out[key].map(meta[spec.relation][spec.field])
                if spec.aggregation == 'raw_category':
                    values = values.astype('string').map(
                        self.encodings[f'{spec.relation}.{spec.field}']).fillna(-1)
            elif spec.relation in ['user', 'item', 'pair']:
                agg = 'max' if spec.aggregation == 'recency' else spec.aggregation
                series = aggregate(spec.relation, spec.window_days, agg,
                                   r.time if agg == 'max' else spec.field)
                if spec.relation == 'pair':
                    idx = pd.MultiIndex.from_frame(out[[r.user, r.item]])
                    values = pd.Series(series.reindex(idx, fill_value=0).to_numpy())
                else:
                    values = out[getattr(r, spec.relation)].map(series)
                    if spec.aggregation == 'count':
                        values = values.fillna(0)
                    elif spec.aggregation == 'recency':
                        values = (cutoff - values).dt.total_seconds() / 86400
            elif spec.relation == 'numeric_taste':
                key = spec.field
                if key not in cache:
                    joined = history[[r.user, r.item]].merge(meta['item'][[key]], on=r.item, how='left', validate='many_to_one')
                    cache[key] = joined.groupby(r.user)[key].mean()
                values = (out[r.item].map(meta['item'][key]) - out[r.user].map(cache[key])).abs()
            else:
                key = (spec.field, spec.window_days, spec.aggregation)
                if key not in relations:
                    joined = windows[spec.window_days].merge(edges[spec.field], on=r.item, how='inner')
                    g = joined.groupby([r.user, '_token'], sort=False)
                    relations[key] = (g.size() if spec.aggregation == 'share' else
                                      (cutoff - g[r.time].max()).dt.total_seconds() / 86400)
                query = queries[spec.field]
                idx = pd.MultiIndex.from_frame(query[[r.user, '_token']])
                mapped = relations[key].reindex(idx).to_numpy(dtype=float)
                if spec.aggregation == 'share':
                    mapped = np.nan_to_num(mapped, nan=0)
                values = pd.Series(mapped).groupby(query['_row'].to_numpy()).agg(spec.reduction).reindex(range(len(out)))
                if spec.aggregation == 'share':
                    denom = out[r.user].map(aggregate('user', spec.window_days, 'count')).fillna(0).to_numpy()
                    values = values / np.maximum(denom, 1)
                values = values.where(known[spec.field])
            out[spec.name] = np.asarray(values, dtype=float)
        if np.isinf(out[self.columns].to_numpy()).any():
            raise ValueError('Infinite generated feature values')
        return out


# Budget settings used by the unified study for V1 (windows are fixed to 7 and 30 days).
SETTINGS = {'max_features': 64, 'max_relations': 5, 'max_event_numeric': 2,
            'max_metadata_numeric': 4, 'max_categories': 32, 'max_tokens': 4, 'max_vocab': 8192}
