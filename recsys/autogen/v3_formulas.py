"""Typed, bounded expression synthesis and cutoff-safe compilation from raw tables.

No eval/exec, manual feature list or required V1/V2 feature base. A schema profiler
binds fields; table operators compose into aggregates and scalar expressions.
"""
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from .v1_schema import FeatureGenerator, tokens


def canonical(expr):
    return json.dumps(expr, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def name(expr):
    return 'expr__' + hashlib.sha256(canonical(expr).encode()).hexdigest()[:16]


def nodes(expr):
    return 1 + sum(nodes(v) for v in expr.values() if isinstance(v, dict))


def depth(expr):
    return 1 + max([depth(v) for v in expr.values() if isinstance(v, dict)] or [0])


def window(days, offset=0):
    return {'op': 'events'} if days is None else {'op': 'window', 'days': days, 'offset': offset, 'input': {'op': 'events'}}


def aggregate(entity, source, function='count', field=None, reduction='mean'):
    return {'op': 'aggregate', 'entity': entity, 'function': function, 'field': field,
            'reduction': reduction, 'input': source}


def ratio(a, b):
    return {'op': 'divide', 'left': a, 'right': b}


def add_one(expr):
    return {'op': 'add', 'left': expr, 'right': {'op': 'constant', 'value': 1.}}


def attribute_lift(field, source, prior_strength=10., reduction='mean'):
    return {'op': 'attribute_lift', 'field': field, 'prior_strength': prior_strength,
            'reduction': reduction, 'input': source}


def segment_item_lift(field, source, prior_strength=10.):
    return {'op': 'segment_item_lift', 'field': field,
            'prior_strength': prior_strength, 'input': source}


class FormulaGenerator:
    def __init__(self, max_seeds=128, max_mutations=64, max_nodes=24, max_depth=10):
        self.settings = dict(max_seeds=max_seeds, max_mutations=max_mutations,
                             max_nodes=max_nodes, max_depth=max_depth)

    def fit(self, events, cutoff, users, items):
        # Reuse only schema/type/key inference, never its hand-authored recipes.
        profiler = FeatureGenerator().fit(events, cutoff, users, items)
        self.profiler = profiler; self.roles = profiler.roles
        history = profiler.history(events, cutoff)
        self.numeric = list(profiler.schema['event_numeric'])
        self.thresholds = {f: {str(q): float(history[f].quantile(q)) for q in [.5, .8]}
                           for f in self.numeric}
        self.expressions = []
        for entity in ['user', 'item']:
            for f in [*profiler.schema['numeric'][entity], *profiler.schema['categories'][entity]]:
                self.add({'op': 'metadata', 'entity': entity, 'field': f})
            for days in [None, 3, 7, 10, 14, 30]:
                self.add(aggregate(entity, window(days)))
            for f in self.numeric:
                for days in [None, 7, 14, 30]:
                    self.add(aggregate(entity, window(days), 'mean', f))
                self.add(aggregate(entity, window(None), 'std', f))
            self.add(aggregate(entity, window(None), 'last_age'))
        for attribute in profiler.schema['relations']:
            for days in [None, 7, 14, 30]:
                source = {'op': 'join_attribute', 'field': attribute, 'input': window(days)}
                count = aggregate('user_attribute', source)
                self.add(count)
                self.add(ratio(count, aggregate('user', window(days))))
            for f in self.numeric:
                source = {'op': 'join_attribute', 'field': attribute, 'input': window(None)}
                self.add(aggregate('user_attribute', source, 'mean', f))
        for f in profiler.schema['numeric']['item']:
            source = {'op': 'join_numeric', 'field': f, 'input': window(None)}
            self.add({'op': 'abs_diff', 'left': {'op': 'metadata', 'entity': 'item', 'field': f},
                      'right': aggregate('user', source, 'mean', '_meta_value')})
        for f in self.numeric:
            for entity in ['user', 'item']:
                for days in [14, 30]:
                    source = {'op': 'filter', 'field': f, 'quantile': .8, 'input': window(days)}
                    self.add(aggregate(entity, source))
        self.expressions = self.expressions[:self.settings['max_seeds']]
        return self

    def add(self, expr):
        self.validate(expr)
        key = name(expr)
        if key not in {name(e) for e in self.expressions}:
            self.expressions.append(expr)

    def validate(self, expr):
        if nodes(expr) > self.settings['max_nodes'] or depth(expr) > self.settings['max_depth']:
            raise ValueError('Expression complexity exceeds declared bound')
        p = self.profiler.schema
        def check(e):
            op = e.get('op')
            if op == 'events': return ('table', set(self.numeric), None)
            if op in ['window', 'filter', 'join_attribute', 'join_numeric', 'map_log', 'decay']:
                typ, fields, attr = check(e['input'])
                if typ != 'table': raise ValueError('Table operator needs table input')
                if op == 'window':
                    if e['days'] not in [3, 7, 10, 14, 30] or e.get('offset', 0) not in [0, e['days']]:
                        raise ValueError('Invalid window/offset')
                if op == 'filter' and (e['field'] not in self.numeric or str(e['quantile']) not in self.thresholds[e['field']]):
                    raise ValueError('Filter must use a frozen training quantile')
                if op == 'join_attribute':
                    if attr is not None or e['field'] not in p['relations']:
                        raise ValueError('Only one discovered attribute join allowed')
                    attr = e['field']
                if op == 'join_numeric':
                    if e['field'] not in p['numeric']['item'] or '_meta_value' in fields:
                        raise ValueError('Unknown/repeated numeric metadata join')
                    fields = fields | {'_meta_value'}
                if op == 'map_log':
                    if e['field'] not in self.numeric or '_log_value' in fields:
                        raise ValueError('Log mapping needs an original numeric event field')
                    fields = fields | {'_log_value'}
                if op == 'decay':
                    if e['half_life'] not in [7, 14, 30] or '_weight' in fields:
                        raise ValueError('Invalid/repeated temporal decay')
                    fields = fields | {'_weight'}
                return ('table', fields, attr)
            if op == 'segment_item_lift':
                typ, _, attr = check(e['input'])
                alpha = e['prior_strength']
                if (typ != 'table' or attr is not None or e['field'] not in p['categories']['user'] or
                        not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or
                        not np.isfinite(alpha) or alpha <= 0):
                    raise ValueError('Invalid segment item lift')
                return ('scalar', set(), None)
            if op == 'attribute_lift':
                typ, _, attr = check(e['input'])
                alpha = e['prior_strength']
                if (typ != 'table' or attr is not None or e['field'] not in p['relations'] or
                        not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or
                        not np.isfinite(alpha) or alpha <= 0 or
                        e['reduction'] not in ['mean', 'max', 'min']):
                    raise ValueError('Invalid attribute lift')
                return ('scalar', set(), None)
            if op == 'aggregate':
                typ, fields, attr = check(e['input'])
                if typ != 'table' or e['entity'] not in ['user', 'item', 'user_attribute', 'global']:
                    raise ValueError('Invalid aggregate entity')
                if e['entity'] == 'user_attribute' and attr is None:
                    raise ValueError('Attribute projection needs an attribute join')
                if e['function'] not in ['count', 'mean', 'std', 'last_age', 'sum', 'valid_count']:
                    raise ValueError('Invalid aggregation')
                if e['function'] in ['mean', 'std', 'sum', 'valid_count'] and e['field'] not in fields:
                    raise ValueError('Aggregation field absent from source')
                if e['reduction'] not in ['mean', 'max', 'min']:
                    raise ValueError('Invalid candidate-token reduction')
                return ('scalar', set(), None)
            if op == 'metadata':
                entity = e['entity']
                if entity not in ['user', 'item'] or e['field'] not in p['numeric'][entity] + p['categories'][entity]:
                    raise ValueError('Unknown metadata field')
                return ('scalar', set(), None)
            if op == 'constant':
                if e['value'] not in [1., 10.]: raise ValueError('Unsupported constant')
                return ('scalar', set(), None)
            if op == 'log1p_clip0':
                if check(e['input'])[0] != 'scalar': raise ValueError('Unary input must be scalar')
                return ('scalar', set(), None)
            if op in ['divide', 'add', 'abs_diff', 'multiply']:
                if any(check(e[k])[0] != 'scalar' for k in ['left', 'right']):
                    raise ValueError('Arithmetic needs scalar inputs')
                return ('scalar', set(), None)
            raise ValueError('Unknown expression operation')
        if check(expr)[0] != 'scalar':
            raise ValueError('Final expression must produce a feature vector')

    @property
    def columns(self): return [name(e) for e in self.expressions]

    @property
    def categorical(self):
        return [name(e) for e in self.expressions if e['op'] == 'metadata' and
                e['field'] in self.profiler.schema['categories'][e['entity']]]

    def mutate(self, ranked_names):
        original = list(self.expressions)
        lookup = {name(e): e for e in original}
        parents = [lookup[n] for n in ranked_names if n in lookup and n not in self.categorical][:12]
        candidates = []
        for parent in parents:
            candidates.append({'op': 'log1p_clip0', 'input': parent})
            if parent['op'] != 'aggregate': continue
            for days in [3, 7, 10, 14, 30]:
                e = copy.deepcopy(parent)
                def rewindow(source):
                    if source['op'] in ['window', 'events']: return window(days)
                    return {**source, 'input': rewindow(source['input'])}
                e['input'] = rewindow(e['input']); candidates.append(e)
            for f in self.numeric:
                for q in [.5, .8]:
                    e = copy.deepcopy(parent)
                    e['input'] = {'op': 'filter', 'field': f, 'quantile': q, 'input': e['input']}
                    candidates.append(e)
            if parent['entity'] == 'user_attribute':
                candidates.extend([{**parent, 'reduction': reduction} for reduction in ['max', 'min']])
            if parent['function'] == 'count':
                candidates.extend([{**parent, 'function': fn, 'field': f}
                                   for f in self.numeric for fn in ['mean', 'std']])
                if parent['entity'] in ['user', 'item']:
                    for days in [7, 14, 30]:
                        a = aggregate(parent['entity'], window(days))
                        b = aggregate(parent['entity'], window(days, offset=days))
                        candidates.append(ratio(add_one(a), add_one(b)))
        # Round-robin parent mutations by operation, so a window sweep cannot
        # consume the entire budget before filters or arithmetic get a chance.
        candidates.sort(key=lambda e: (depth(e), canonical(e)))
        buckets = {}
        for e in candidates:
            op = e['op'] if e['op'] != 'aggregate' else ('filter' if e['input']['op'] == 'filter' else e['function'])
            buckets.setdefault(op, []).append(e)
        start = len(self.expressions)
        while any(buckets.values()) and len(self.expressions) - start < self.settings['max_mutations']:
            for op in sorted(buckets):
                if not buckets[op]: continue
                e = buckets[op].pop(0)
                try: self.add(e)
                except ValueError: continue
                if len(self.expressions) - start >= self.settings['max_mutations']: break
        return [name(e) for e in self.expressions[start:]]

    def export(self):
        return {'version': 3, 'settings': self.settings, 'schema_profile': self.profiler.schema,
                'thresholds': self.thresholds, 'expressions': [{'name': name(e), 'ast': e} for e in self.expressions]}

    def enrich(self, max_extra=64):
        """Compose missing operator combinations, binding fields from the schema.

        These are candidates, never a required base. In particular, log maps
        happen before aggregation, unlike monotone logs of existing aggregates.
        """
        start = len(self.expressions)
        candidates = []
        for entity in ['user', 'item']:
            for field in self.numeric:
                source = {'op': 'map_log', 'field': field, 'input': window(None)}
                candidates.append(aggregate(entity, source, 'mean', '_log_value'))
        for field in self.numeric:
            for days in [None, 7, 14, 30]:
                source = window(days)
                prior = aggregate('global', source, 'mean', field)
                numerator = {'op': 'add', 'left': aggregate('item', source, 'sum', field),
                             'right': {'op': 'multiply', 'left': {'op': 'constant', 'value': 10.}, 'right': prior}}
                denominator = {'op': 'add', 'left': aggregate('item', source, 'valid_count', field),
                               'right': {'op': 'constant', 'value': 10.}}
                candidates.append(ratio(numerator, denominator))
        for field in self.profiler.schema['relations']:
            source = {'op': 'join_attribute', 'field': field, 'input': window(None)}
            candidates.append(aggregate('user_attribute', source, 'last_age', reduction='min'))
            for days in [None, 14, 30]:
                source = {'op': 'join_attribute', 'field': field, 'input': window(days)}
                candidates.append(ratio(aggregate('user_attribute', source, reduction='max'),
                                        aggregate('user', window(days))))
            for half_life in [7, 14]:
                source = {'op': 'decay', 'half_life': half_life, 'input': window(None)}
                joined = {'op': 'join_attribute', 'field': field, 'input': source}
                candidates.append(ratio(aggregate('user_attribute', joined, 'sum', '_weight'),
                                        aggregate('user', source, 'sum', '_weight')))
        candidates.extend(attribute_lift(field, window(None))
                          for field in self.profiler.schema['relations'])
        for expr in candidates:
            if len(self.expressions) - start >= max_extra: break
            self.add(expr)
        return self.columns[start:]

    @classmethod
    def from_schema(cls, schema):
        obj = cls(**schema['settings'])
        obj.profiler = FeatureGenerator.from_schema(schema['schema_profile']); obj.roles = obj.profiler.roles
        obj.numeric = list(obj.profiler.schema['event_numeric']); obj.thresholds = schema['thresholds']
        obj.expressions = []
        for rec in schema['expressions']:
            if name(rec['ast']) != rec['name']: raise ValueError('Expression hash mismatch')
            obj.add(rec['ast'])
        return obj

    def transform(self, events, cutoff, candidates, users, items, columns=None):
        cutoff = pd.Timestamp(cutoff)
        if pd.Timestamp(cutoff) < pd.Timestamp(self.profiler.schema['fit_cutoff']):
            raise ValueError('Cannot apply a schema earlier than its fit cutoff')
        r = self.roles; history = self.profiler.history(events, cutoff)
        out = candidates[[r.user, r.item]].copy().reset_index(drop=True)
        if out.duplicated().any(): raise ValueError('Duplicate candidate pairs')
        for key in [r.user, r.item]:
            if not set(out[key]) <= set(history[key]): raise ValueError('Candidates must be historical')
        meta = {'user': self.profiler._metadata(users, r.user, history[r.user]).set_index(r.user),
                'item': self.profiler._metadata(items, r.item, history[r.item]).set_index(r.item)}
        cache, edge_cache, query_cache = {}, {}, {}
        def edges(field):
            if field not in edge_cache:
                lists = meta['item'][field].map(tokens)
                edge_cache[field] = lists.rename('_token').reset_index().explode('_token').dropna(subset=['_token'])
            return edge_cache[field]
        def evaluate(e):
            k = canonical(e)
            if k in cache: return cache[k]
            op = e['op']
            if op == 'events': value = (history, None)
            elif op in ['window', 'filter', 'join_attribute', 'join_numeric', 'map_log', 'decay']:
                frame, attr = evaluate(e['input'])
                if op == 'window':
                    upper = cutoff - pd.Timedelta(days=e.get('offset', 0))
                    value = (frame.loc[(frame[r.time] >= upper - pd.Timedelta(days=e['days'])) & (frame[r.time] < upper)], attr)
                elif op == 'filter':
                    value = (frame.loc[frame[e['field']] >= self.thresholds[e['field']][str(e['quantile'])]], attr)
                elif op == 'join_attribute':
                    value = (frame.merge(edges(e['field']), on=r.item, how='inner'), e['field'])
                elif op == 'map_log':
                    value = (frame.assign(_log_value=np.log1p(frame[e['field']].clip(lower=0))), attr)
                elif op == 'decay':
                    age = (cutoff - frame[r.time]).dt.total_seconds() / 86400
                    value = (frame.assign(_weight=np.exp2(-age / e['half_life'])), attr)
                else:
                    values = meta['item'][e['field']].rename('_meta_value')
                    value = (frame.merge(values, on=r.item, how='left', validate='many_to_one'), attr)
            elif op == 'segment_item_lift':
                frame, _ = evaluate(e['input'])
                field, alpha = e['field'], e['prior_strength']
                categories = meta['user'][field].astype('string')
                joined = frame.assign(_segment=frame[r.user].map(categories))
                segment = out[r.user].map(categories)
                index = pd.MultiIndex.from_arrays([segment, out[r.item]])
                counts = joined.groupby(['_segment', r.item]).size().reindex(index, fill_value=0).to_numpy(dtype=float)
                totals = segment.map(joined.groupby('_segment').size()).fillna(0).to_numpy(dtype=float)
                probability = frame.groupby(r.item).size() / len(frame) if len(frame) else pd.Series(dtype=float)
                p = out[r.item].map(probability).to_numpy(dtype=float)
                value = np.divide(counts + alpha * p, (totals + alpha) * p,
                                  out=np.full(len(out), np.nan), where=p > 0)
                value[segment.isna().to_numpy()] = np.nan
            elif op == 'attribute_lift':
                frame, _ = evaluate(e['input'])
                field, alpha = e['field'], e['prior_strength']
                joined = frame.merge(edges(field), on=r.item, how='inner')
                if field not in query_cache:
                    query_cache[field] = out.assign(_row=np.arange(len(out))).merge(edges(field), on=r.item, how='left')
                query = query_cache[field]
                index = pd.MultiIndex.from_frame(query[[r.user, '_token']])
                # Normalize each token before candidate-token reduction. Both
                # denominators count source events before token expansion.
                probability = joined.groupby('_token').size() / len(frame) if len(frame) else pd.Series(dtype=float)
                p = query._token.map(probability).to_numpy(dtype=float)
                count = joined.groupby([r.user, '_token']).size().reindex(index, fill_value=0).to_numpy(dtype=float)
                total = query[r.user].map(frame.groupby(r.user).size()).fillna(0).to_numpy(dtype=float)
                mapped = np.divide(count + alpha * p, (total + alpha) * p,
                                   out=np.full(len(query), np.nan), where=p > 0)
                value = pd.Series(mapped).groupby(query._row.to_numpy()).agg(e['reduction']).reindex(range(len(out))).to_numpy()
            elif op == 'aggregate':
                frame, attr = evaluate(e['input']); entity, fn = e['entity'], e['function']
                if entity == 'global':
                    if fn == 'count': result = len(frame)
                    elif fn == 'last_age': result = (cutoff - frame[r.time].max()).total_seconds() / 86400
                    elif fn == 'valid_count': result = frame[e['field']].count()
                    elif fn == 'std': result = frame[e['field']].std(ddof=0)
                    else: result = getattr(frame[e['field']], fn)()
                    value = np.full(len(out), result)
                    cache[k] = value
                    return value
                by = [r.user, '_token'] if entity == 'user_attribute' else [getattr(r, entity)]
                group = frame.groupby(by, sort=False)
                if fn == 'count': series = group.size()
                elif fn == 'valid_count': series = group[e['field']].count()
                elif fn == 'sum': series = group[e['field']].sum()
                elif fn == 'last_age': series = (cutoff - group[r.time].max()).dt.total_seconds() / 86400
                elif fn == 'std': series = group[e['field']].std(ddof=0)
                else: series = group[e['field']].mean()
                if entity != 'user_attribute':
                    value = out[getattr(r, entity)].map(series).to_numpy(dtype=float)
                    if fn in ['count', 'valid_count', 'sum']: value = np.nan_to_num(value, nan=0)
                else:
                    if attr not in query_cache:
                        query_cache[attr] = out.assign(_row=np.arange(len(out))).merge(edges(attr), on=r.item, how='left')
                    query = query_cache[attr]
                    index = pd.MultiIndex.from_frame(query[[r.user, '_token']])
                    mapped = series.reindex(index).to_numpy(dtype=float)
                    if fn in ['count', 'valid_count', 'sum']: mapped = np.nan_to_num(mapped, nan=0)
                    value = pd.Series(mapped).groupby(query._row.to_numpy()).agg(e['reduction']).reindex(range(len(out))).to_numpy(copy=True)
                    value[~out[r.item].isin(edges(attr)[r.item]).to_numpy()] = np.nan
            elif op == 'metadata':
                values = out[getattr(r, e['entity'])].map(meta[e['entity']][e['field']])
                if e['field'] in self.profiler.schema['categories'][e['entity']]:
                    values = values.astype('string').map(self.profiler.encodings[f"{e['entity']}.{e['field']}"]).fillna(-1)
                value = values.to_numpy(dtype=float)
            elif op == 'constant': value = np.full(len(out), e['value'])
            elif op == 'log1p_clip0': value = np.log1p(np.maximum(evaluate(e['input']), 0))
            else:
                a, b = evaluate(e['left']), evaluate(e['right'])
                if op == 'add': value = a + b
                elif op == 'abs_diff': value = np.abs(a - b)
                elif op == 'multiply': value = a * b
                else:
                    value = np.divide(a, b, out=np.full(len(out), np.nan), where=b != 0)
                    value = np.where((b == 0) & np.isfinite(a), 0., value)
            cache[k] = value
            return value
        lookup = {name(e): e for e in self.expressions}
        columns = self.columns if columns is None else columns
        feature_values = {}
        for col in columns:
            if col not in lookup: raise ValueError('Unknown expression column')
            feature_values[col] = evaluate(lookup[col])
        out = pd.concat([out, pd.DataFrame(feature_values)], axis=1)
        if np.isinf(out[columns].to_numpy()).any(): raise ValueError('Infinite formula feature values')
        return out
