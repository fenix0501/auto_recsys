"""Additive schema-driven grammar, isolated from frozen v1 numerical code."""
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd

from .v1_schema import FeatureGenerator, FeatureBudgetWarning, Roles, tokens


@dataclass(frozen=True)
class ExtraSpec:
    entity: str
    operation: str
    attribute: str | None = None
    measure: str | None = None
    window_days: int | None = None

    @property
    def name(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]
        parts = [self.entity, self.operation, self.attribute or 'rows', self.measure or 'value',
                 str(self.window_days or 'all')]
        return 'v2__' + '__'.join(re.sub('[^a-zA-Z0-9_]', '_', x) for x in parts) + '__' + digest


class FeatureGeneratorV2:
    def __init__(self, families=(), windows=(14, 30), max_features=100, prior_strength=10., **base_settings):
        if not set(families) <= {'dynamics', 'affinity', 'segments'}:
            raise ValueError('Unknown feature family')
        if max_features < 64 or prior_strength <= 0:
            raise ValueError('Need a base feature budget and a positive prior')
        if isinstance(base_settings.get('roles'), Roles):
            base_settings['roles'] = asdict(base_settings['roles'])
        self.settings = {'families': sorted(set(families)), 'windows': list(windows),
                         'max_features': max_features, 'prior_strength': prior_strength,
                         'base_settings': base_settings}

    def fit(self, events, cutoff, users, items):
        cfg = self.settings
        base_settings = dict(cfg['base_settings'])
        if isinstance(base_settings.get('roles'), dict):
            base_settings['roles'] = Roles(**base_settings['roles'])
        self.base = FeatureGenerator(windows=tuple(cfg['windows']), max_features=64,
                                     **base_settings).fit(events, cutoff, users, items)
        history = self.base.history(events, cutoff)
        r = self.base.roles
        numeric = self.base.schema['event_numeric']
        short = min(cfg['windows'])
        specs = []
        if 'dynamics' in cfg['families']:
            for entity in ['user', 'item']:
                specs.append(ExtraSpec(entity, 'trend', window_days=short))
                for field in numeric:
                    if history[field].dropna().ge(0).all():
                        specs.append(ExtraSpec(entity, 'log_mean', measure=field))
                    specs.append(ExtraSpec(entity, 'std', measure=field))
            for field in numeric:
                for window in [None, short]:
                    specs.append(ExtraSpec('item', 'smoothed_mean', measure=field, window_days=window))
        if 'affinity' in cfg['families']:
            for field in self.base.schema['relations']:
                specs.extend([ExtraSpec('attribute', 'decay_share', attribute=field, window_days=short),
                              ExtraSpec('attribute', 'smoothed_lift', attribute=field)])
                specs.extend(ExtraSpec('attribute', 'conditional_mean', attribute=field, measure=n)
                             for n in numeric)
        if 'segments' in cfg['families']:
            specs.extend(ExtraSpec('segment', 'smoothed_item_lift', attribute=f)
                         for f in self.base.schema['categories']['user'])
        budget = max(cfg['max_features'] - len(self.base.columns), 0)
        self.extra_specs = specs[:budget]
        if len(specs) > budget:
            warnings.warn('V2 feature budget truncated extra recipes; see schema.truncated_recipes',
                          FeatureBudgetWarning, stacklevel=2)
        self.schema = {'version': 2, 'settings': cfg, 'base_schema': self.base.schema,
                       'extra_recipes': [asdict(s) | {'name': s.name} for s in self.extra_specs],
                       'truncated_recipes': [asdict(s) | {'name': s.name} for s in specs[budget:]]}
        return self._bind()

    def _bind(self):
        self.roles = self.base.roles
        self.columns = [*self.base.columns, *[s.name for s in self.extra_specs]]
        self.categorical = list(self.base.categorical)
        return self

    @classmethod
    def from_schema(cls, schema):
        cfg = schema['settings']
        obj = cls(**{k: v for k, v in cfg.items() if k != 'base_settings'}, **cfg['base_settings'])
        obj.base = FeatureGenerator.from_schema(schema['base_schema'])
        obj.extra_specs = [ExtraSpec(**{k: v for k, v in s.items() if k != 'name'})
                           for s in schema['extra_recipes']]
        if [s.name for s in obj.extra_specs] != [s['name'] for s in schema['extra_recipes']]:
            raise ValueError('V2 recipe names do not match exported schema')
        obj.schema = schema
        return obj._bind()

    def transform(self, events, cutoff, candidates, users, items):
        out = self.base.transform(events, cutoff, candidates, users, items)
        history = self.base.history(events, cutoff)
        r, alpha = self.roles, self.settings['prior_strength']
        user_meta = self.base._metadata(users, r.user, history[r.user]).set_index(r.user)
        item_meta = self.base._metadata(items, r.item, history[r.item]).set_index(r.item)
        cache = {}
        def window(days):
            return history if days is None else history.loc[history[r.time] >= cutoff - pd.Timedelta(days=days)]
        for spec in self.extra_specs:
            if spec.entity in ['user', 'item']:
                key = getattr(r, spec.entity)
                if spec.operation == 'trend':
                    days = spec.window_days
                    recent = window(days).groupby(key).size()
                    prev = history.loc[(history[r.time] >= cutoff - pd.Timedelta(days=2 * days)) &
                                       (history[r.time] < cutoff - pd.Timedelta(days=days))].groupby(key).size()
                    values = (out[key].map(recent).fillna(0) + 1) / (out[key].map(prev).fillna(0) + 1)
                elif spec.operation == 'log_mean':
                    means = history.assign(_value=np.log1p(history[spec.measure].clip(lower=0))).groupby(key)._value.mean()
                    values = out[key].map(means)
                elif spec.operation == 'std':
                    values = out[key].map(history.groupby(key)[spec.measure].std(ddof=0))
                else:
                    frame = window(spec.window_days)
                    g = frame.groupby(key)[spec.measure]
                    prior = frame[spec.measure].mean()
                    totals = out[key].map(g.sum()).fillna(0)
                    counts = out[key].map(g.count()).fillna(0)
                    values = (totals + alpha * prior) / (counts + alpha)
            elif spec.entity == 'attribute':
                field = spec.attribute
                if field not in cache:
                    lists = item_meta[field].map(tokens)
                    edges = lists.rename('_token').reset_index().explode('_token').dropna(subset=['_token'])
                    joined = history.merge(edges, on=r.item, how='inner')
                    query = out[[r.user, r.item]].assign(_row=np.arange(len(out))).merge(edges, on=r.item, how='left')
                    cache[field] = (joined, query)
                joined, query = cache[field]
                index = pd.MultiIndex.from_frame(query[[r.user, '_token']])
                group = joined.groupby([r.user, '_token'], sort=False)
                if spec.operation == 'conditional_mean':
                    mapped = group[spec.measure].mean().reindex(index).to_numpy(dtype=float)
                elif spec.operation == 'decay_share':
                    age = (cutoff - joined[r.time]).dt.total_seconds() / 86400
                    weighted = joined.assign(_weight=np.exp2(-age / spec.window_days))
                    totals = weighted.groupby([r.user, '_token'])._weight.sum()
                    orig_age = (cutoff - history[r.time]).dt.total_seconds() / 86400
                    denom = history.assign(_weight=np.exp2(-orig_age / spec.window_days)).groupby(r.user)._weight.sum()
                    mapped = totals.reindex(index, fill_value=0).to_numpy(dtype=float)
                    mapped = mapped / query[r.user].map(denom).to_numpy(dtype=float)
                else:
                    probability = joined.groupby('_token').size() / len(history)
                    p = query._token.map(probability).to_numpy(dtype=float)
                    count = group.size().reindex(index, fill_value=0).to_numpy(dtype=float)
                    total = query[r.user].map(history.groupby(r.user).size()).to_numpy(dtype=float)
                    mapped = np.divide(count + alpha * p, (total + alpha) * p,
                                       out=np.full(len(query), np.nan), where=p > 0)
                # Mean over distinct candidate tokens; missing metadata stays NaN.
                values = pd.Series(mapped).groupby(query._row.to_numpy()).mean().reindex(range(len(out)))
                missing = ~out[r.item].isin(joined[r.item])
                values = values.where(~missing.to_numpy())
            else:
                field = spec.attribute
                cats = user_meta[field].astype('string')
                joined = history.assign(_segment=history[r.user].map(cats))
                counts = joined.groupby(['_segment', r.item]).size()
                totals = joined.groupby('_segment').size()
                candidate_segment = out[r.user].map(cats)
                index = pd.MultiIndex.from_arrays([candidate_segment, out[r.item]])
                count = counts.reindex(index, fill_value=0).to_numpy(dtype=float)
                n = candidate_segment.map(totals).fillna(0).to_numpy(dtype=float)
                p = out[r.item].map(history.groupby(r.item).size()).to_numpy(dtype=float) / len(history)
                values = pd.Series((count + alpha * p) / ((n + alpha) * p)).where(candidate_segment.notna())
            out[spec.name] = np.asarray(values, dtype=float)
        if np.isinf(out[self.columns].to_numpy()).any():
            raise ValueError('Infinite V2 generated values')
        return out


# Family variants searched by the unified study (crossed with the window pairs).
VARIANTS = {'baseline': (), 'dynamics': ('dynamics',), 'affinity': ('affinity',),
            'segments': ('segments',), 'all': ('dynamics', 'affinity', 'segments')}
