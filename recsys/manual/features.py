"""Shared historical feature builder and a bounded 48-feature grammar."""
from __future__ import annotations
from time import perf_counter

import numpy as np
import pandas as pd

from ..common.candidates import COMMON_FEATURES
from ..common.contracts import days_elapsed, historical_events, validate_metadata

# Fixed KION taxonomies, not vocabulary learned from future-only metadata rows.
# Order preserves the codes used by the initial experiment. Unknown values are -1.
CATEGORY_ENCODINGS = {
    'age': {v:i for i,v in enumerate(['age_18_24','age_25_34','age_35_44','age_45_54','age_55_64','age_65_inf'])},
    'income': {v:i for i,v in enumerate(['income_0_20','income_150_inf','income_20_40','income_40_60','income_60_90','income_90_150'])},
    'sex': {'Ж':0,'М':1}, 'content_type': {'film':0,'series':1},
}

CORE_FEATURES = [
    'u_age', 'u_income', 'u_sex', 'u_kids', 'u_count7', 'u_count30',
    'u_mean_pct_all', 'u_recency', 'u_series_share', 'i_content_type',
    'i_year', 'i_age_rating', 'i_count7', 'i_count30', 'i_mean_pct_all',
    'i_trend7', 'ui_genre_share', 'ui_type_share',
]
STRONG_FEATURES = [
    'u_count3', 'u_count14', 'u_count90', 'u_countall', 'u_mean_logdur',
    'u_pct_std', 'u_completion80', 'u_active_span', 'u_genre_entropy',
    'u_activity_trend7', 'i_count1', 'i_count3', 'i_count14', 'i_countall',
    'i_recency', 'i_age_days', 'i_smoothed_pct', 'i_mean_logdur',
    'i_completion80', 'i_trend3', 'i_genre_count',
    'i_pop_share3', 'i_pop_share7', 'i_pop_share30',
    'ui_genre_share7', 'ui_genre_share14', 'ui_genre_share30',
    'ui_genre_max_share7', 'ui_genre_max_share14', 'ui_genre_max_share30',
    'ui_genre_max_shareall', 'ui_genre_decay_share14', 'ui_genre_last_seen',
    'ui_genre_completion', 'ui_genre_lastday_overlap', 'ui_genre_last3days_overlap',
    'ui_genre_novelty', 'ui_genre_lift', 'ui_type_share7', 'ui_type_share30',
    'ui_country_share', 'ui_country_last_seen', 'ui_director_share',
    'ui_director_last_seen', 'ui_release_year_gap', 'ui_age_rating_gap',
    'i_agegroup_lift', 'i_gender_lift',
]
MANUAL_FEATURES = [*CORE_FEATURES, *STRONG_FEATURES]
CATEGORICAL_FEATURES = ['u_age', 'u_income', 'u_sex', 'i_content_type']


def genre_table(items):
    frame = items[['item_id', 'genres']].copy()
    frame['genre'] = frame['genres'].fillna('').str.split(',').map(
        lambda values: [v.strip() for v in values if v.strip()])
    return frame[['item_id', 'genre']].explode('genre').dropna(subset=['genre']).drop_duplicates().reset_index(drop=True)


def build_features(events, cutoff, candidates, users, items, config):
    history = historical_events(events, cutoff)
    start = perf_counter()
    validate_metadata(users, items)
    if not set(candidates['user_id']).issubset(set(history['user_id'])):
        raise ValueError('Feature candidates must have historical users')
    if not set(candidates['item_id']).issubset(set(history['item_id'])):
        raise ValueError('Feature candidates must have items observed before cutoff')
    if candidates.duplicated(['user_id','item_id']).any():
        raise ValueError('Duplicate candidate user-item pairs')
    if not config['windows'] or any(d<=0 for d in config['windows']):
        raise ValueError('Feature windows must be positive')
    # Targets and arbitrary candidate columns cannot be propagated into features.
    out = candidates[['user_id','item_id',*COMMON_FEATURES]].copy().reset_index(drop=True)
    by_user = history.groupby('user_id', sort=False)
    by_item = history.groupby('item_id', sort=False)
    user_meta = users.set_index('user_id')
    item_meta = items.set_index('item_id')
    for column, name in [('age', 'u_age'), ('income', 'u_income'), ('sex', 'u_sex')]:
        vals = user_meta[column].astype('string')
        out[name] = out['user_id'].map(vals.map(CATEGORY_ENCODINGS[column])).fillna(-1)
    out['u_kids'] = out['user_id'].map(user_meta['kids_flg'])
    content = item_meta['content_type'].astype('string')
    out['i_content_type'] = out['item_id'].map(content.map(CATEGORY_ENCODINGS['content_type'])).fillna(-1)
    out['i_year'] = out['item_id'].map(item_meta['release_year'])
    out['i_age_rating'] = out['item_id'].map(item_meta['age_rating'])
    for days in [7, 30]:
        window = history[history['last_watch_dt'] >= cutoff-pd.Timedelta(days=days)]
        out[f'u_count{days}'] = out['user_id'].map(window.groupby('user_id').size()).fillna(0)
        out[f'i_count{days}'] = out['item_id'].map(window.groupby('item_id').size()).fillna(0)
    out['u_mean_pct_all'] = out['user_id'].map(by_user['watched_pct'].mean())
    out['i_mean_pct_all'] = out['item_id'].map(by_item['watched_pct'].mean())
    recency = (cutoff - by_user['last_watch_dt'].max()).dt.total_seconds() / 86400
    out['u_recency'] = out['user_id'].map(recency)
    hist_types = history[['user_id', 'item_id']].copy()
    hist_types['content_type'] = hist_types['item_id'].map(content)
    hist_types['is_series'] = (hist_types['content_type'] == 'series').astype('float32')
    out['u_series_share'] = out['user_id'].map(hist_types.groupby('user_id')['is_series'].mean())
    prev = history[(history['last_watch_dt'] >= cutoff-pd.Timedelta(days=14)) &
                   (history['last_watch_dt'] < cutoff-pd.Timedelta(days=7))]
    prev_count = out['item_id'].map(prev.groupby('item_id').size()).fillna(0)
    out['i_trend7'] = (out['i_count7'] + 1) / (prev_count + 1)
    type_counts = hist_types.groupby(['user_id', 'content_type']).size()
    type_keys = pd.MultiIndex.from_arrays([out['user_id'], out['item_id'].map(content)])
    user_counts = out['user_id'].map(by_user.size()).to_numpy()
    out['ui_type_share'] = type_counts.reindex(type_keys, fill_value=0).to_numpy() / user_counts

    genres = genre_table(items)
    hist_genres = history.merge(genres, on='item_id', how='inner', validate='many_to_many')
    # One source row contributes once to each distinct genre, never twice to a genre.
    hist_genres = hist_genres.drop_duplicates(['user_id', 'item_id', 'last_watch_dt', 'genre'])
    pair_genres = out[['user_id', 'item_id']].assign(_row=np.arange(len(out))).merge(
        genres, on='item_id', how='left', validate='many_to_many')
    genre_keys = pd.MultiIndex.from_arrays([pair_genres['user_id'], pair_genres['genre']])

    def map_genres(series, count=False, reduction='mean'):
        mapped = series.reindex(genre_keys).to_numpy(dtype=float)
        if count:
            mapped = np.nan_to_num(mapped, nan=0)
        grouped = pd.Series(mapped).groupby(pair_genres['_row'], sort=True)
        return grouped.agg(reduction).reindex(np.arange(len(out))).to_numpy()

    genre_counts = hist_genres.groupby(['user_id', 'genre']).size()
    out['ui_genre_share'] = map_genres(genre_counts, count=True) / user_counts
    # Strong FE: recent demand, confidence-aware completion, and personalized
    # metadata affinities. All quantities are built from the cutoff-filtered history.
    total_user = by_user.size()
    total_item = by_item.size()
    for days in [1, 3, 7, 14, 30, 90]:
        w = history[history['last_watch_dt'] >= cutoff-pd.Timedelta(days=days)]
        if days in [3, 14, 90]:
            out[f'u_count{days}'] = out['user_id'].map(w.groupby('user_id').size()).fillna(0)
        if days in [1, 3, 14]:
            out[f'i_count{days}'] = out['item_id'].map(w.groupby('item_id').size()).fillna(0)
        if days in [3, 7, 30]:
            ic = out['item_id'].map(w.groupby('item_id').size()).fillna(0)
            out[f'i_pop_share{days}'] = ic / max(len(w), 1)
    out['u_countall'] = out['user_id'].map(total_user)
    out['i_countall'] = out['item_id'].map(total_item)
    hist_stats = history.assign(logdur=np.log1p(history['total_dur'].clip(lower=0)),
                                completion80=(history['watched_pct'] >= 80).astype(float)
                                .where(history['watched_pct'].notna()))
    for key, prefix in [('user_id', 'u'), ('item_id', 'i')]:
        g = hist_stats.groupby(key)
        out[f'{prefix}_mean_logdur'] = out[key].map(g['logdur'].mean())
        out[f'{prefix}_completion80'] = out[key].map(g['completion80'].mean())
    out['u_pct_std'] = out['user_id'].map(by_user['watched_pct'].std(ddof=0))
    span = days_elapsed(by_user['last_watch_dt'].max()-by_user['last_watch_dt'].min())
    out['u_active_span'] = out['user_id'].map(span)
    previous7 = out['user_id'].map(prev.groupby('user_id').size()).fillna(0)
    out['u_activity_trend7'] = (out['u_count7']+1)/(previous7+1)
    out['i_recency'] = out['item_id'].map(days_elapsed(cutoff-by_item['last_watch_dt'].max()))
    out['i_age_days'] = out['item_id'].map(days_elapsed(cutoff-by_item['last_watch_dt'].min()))
    observed_count = by_item['watched_pct'].count()
    smoothed = (by_item['watched_pct'].sum()+20*history['watched_pct'].mean())/(observed_count+20)
    out['i_smoothed_pct'] = out['item_id'].map(smoothed)
    prior3 = history[(history['last_watch_dt'] >= cutoff-pd.Timedelta(days=6)) &
                     (history['last_watch_dt'] < cutoff-pd.Timedelta(days=3))]
    out['i_trend3'] = (out['i_count3']+1)/(out['item_id'].map(prior3.groupby('item_id').size()).fillna(0)+1)
    out['i_genre_count'] = out['item_id'].map(genres.groupby('item_id').size()).fillna(0)
    genre_total_edges = genre_counts.groupby(level=0).transform('sum')
    prob = genre_counts / genre_total_edges
    entropy = -(prob*np.log(np.maximum(prob, 1e-12))).groupby(level=0).sum()
    out['u_genre_entropy'] = out['user_id'].map(entropy)
    out['ui_genre_max_shareall'] = map_genres(genre_counts, count=True, reduction='max')/user_counts
    out['ui_genre_novelty'] = (map_genres(genre_counts, count=True, reduction='max')==0).astype(float)
    for days in [7, 14, 30]:
        gw = hist_genres[hist_genres['last_watch_dt'] >= cutoff-pd.Timedelta(days=days)]
        gc = gw.groupby(['user_id', 'genre']).size()
        uc = out['user_id'].map(history[history['last_watch_dt'] >= cutoff-pd.Timedelta(days=days)]
                                .groupby('user_id').size()).fillna(0).to_numpy()
        out[f'ui_genre_share{days}'] = map_genres(gc, count=True)/np.maximum(uc, 1)
        out[f'ui_genre_max_share{days}'] = map_genres(gc, count=True, reduction='max')/np.maximum(uc, 1)
    decayed = hist_genres.assign(weight=np.exp2(-days_elapsed(cutoff-hist_genres['last_watch_dt'])/14))
    weighted_genres = decayed.groupby(['user_id', 'genre'])['weight'].sum()
    weighted_user = history.assign(weight=np.exp2(-days_elapsed(cutoff-history['last_watch_dt'])/14))
    weighted_totals = out['user_id'].map(weighted_user.groupby('user_id')['weight'].sum()).to_numpy()
    out['ui_genre_decay_share14'] = map_genres(weighted_genres, count=True)/np.maximum(weighted_totals,1e-8)
    genre_recency = days_elapsed(cutoff-hist_genres.groupby(['user_id','genre'])['last_watch_dt'].max())
    out['ui_genre_last_seen'] = map_genres(genre_recency, reduction='min')
    out['ui_genre_completion'] = map_genres(hist_genres.groupby(['user_id','genre'])['watched_pct'].mean())
    # Dates have daily resolution: context means latest active day(s), not an
    # invented order between several items watched on the same day.
    # Rank all active days, including days on which genre metadata is missing.
    active_days = history.assign(_day=history['last_watch_dt'].dt.normalize())
    active_days['_day_rank'] = active_days.groupby('user_id')['_day'].rank(method='dense', ascending=False)
    latest_ranks = hist_genres[['user_id','item_id']].merge(
        active_days[['user_id','item_id','_day_rank']],on=['user_id','item_id'],
        how='left',validate='many_to_one')['_day_rank']
    for n, name in [(1,'ui_genre_lastday_overlap'), (3,'ui_genre_last3days_overlap')]:
        gc = hist_genres[latest_ranks <= n].groupby(['user_id','genre']).size().clip(upper=1)
        out[name] = map_genres(gc, count=True)
    global_genre_prior = hist_genres.groupby('genre').size()/len(history)
    gp = pair_genres['genre'].map(global_genre_prior).fillna(0).to_numpy()
    gc = genre_counts.reindex(genre_keys).fillna(0).to_numpy()
    uc = pair_genres['user_id'].map(total_user).to_numpy()
    lift = (gc+5*gp)/(uc+5)/np.maximum(gp,1e-8)
    out['ui_genre_lift'] = pd.Series(lift).groupby(pair_genres['_row']).mean().reindex(np.arange(len(out))).to_numpy()
    for days in [7,30]:
        hw = hist_types.loc[history['last_watch_dt'] >= cutoff-pd.Timedelta(days=days)]
        counts = hw.groupby(['user_id','content_type']).size()
        denom = out['user_id'].map(hw.groupby('user_id').size()).fillna(0).to_numpy()
        out[f'ui_type_share{days}'] = counts.reindex(type_keys,fill_value=0).to_numpy()/np.maximum(denom,1)
    for field, name in [('countries','country'),('directors','director')]:
        rel = items[['item_id',field]].copy()
        rel['value'] = rel[field].fillna('').str.split(',')
        rel = rel[['item_id','value']].explode('value')
        rel['value'] = rel['value'].str.strip()
        rel = rel[rel['value'].ne('')].drop_duplicates(['item_id','value'])
        hr = history.merge(rel,on='item_id',how='inner')
        pr = out[['user_id','item_id']].assign(_row=np.arange(len(out))).merge(rel,on='item_id',how='left')
        keys = pd.MultiIndex.from_arrays([pr['user_id'],pr['value']])
        counts = hr.groupby(['user_id','value']).size()
        value = counts.reindex(keys).fillna(0).to_numpy()
        out[f'ui_{name}_share'] = pd.Series(value).groupby(pr['_row']).mean().reindex(np.arange(len(out))).to_numpy()/user_counts
        out.loc[~out['item_id'].isin(rel['item_id']),f'ui_{name}_share'] = np.nan
        recency = days_elapsed(cutoff-hr.groupby(['user_id','value'])['last_watch_dt'].max())
        value = recency.reindex(keys).to_numpy()
        out[f'ui_{name}_last_seen'] = pd.Series(value).groupby(pr['_row']).min().reindex(np.arange(len(out))).to_numpy()
    for field, name in [('release_year','ui_release_year_gap'),('age_rating','ui_age_rating_gap')]:
        preference = history.assign(value=history['item_id'].map(item_meta[field])).groupby('user_id')['value'].mean()
        out[name] = abs(out['item_id'].map(item_meta[field])-out['user_id'].map(preference))
    item_prior = total_item/len(history)
    for field,name in [('age','i_agegroup_lift'),('sex','i_gender_lift')]:
        demographic = user_meta[field].fillna('__unknown__')
        hs = history.assign(segment=history['user_id'].map(demographic).fillna('__unknown__'))
        segment_total = hs.groupby('segment').size()
        segment_items = hs.groupby(['segment','item_id']).size()
        segments = out['user_id'].map(demographic).fillna('__unknown__')
        keys = pd.MultiIndex.from_arrays([segments,out['item_id']])
        count = segment_items.reindex(keys).fillna(0).to_numpy()
        total = segments.map(segment_total).fillna(0).to_numpy()
        prior = out['item_id'].map(item_prior).to_numpy()
        out[name] = (count+20*prior)/(total+20)/np.maximum(prior,1e-8)
    out = out.copy()  # consolidate the many small historical feature columns
    manual_seconds = perf_counter() - start
    auto_start = perf_counter()
    auto_columns = []
    for days in config['windows']:
        floor = cutoff - pd.Timedelta(days=days)
        window = history[history['last_watch_dt'] >= floor]
        genre_window = hist_genres[hist_genres['last_watch_dt'] >= floor]
        for filtered, suffix in [(False, 'all'), (True, 'pct50')]:
            subset = window[window['watched_pct'] >= 50] if filtered else window
            genre_subset = genre_window[genre_window['watched_pct'] >= 50] if filtered else genre_window
            for entity, key in [('user', 'user_id'), ('item', 'item_id'), ('user_genre', None)]:
                groups = (genre_subset.groupby(['user_id', 'genre']) if key is None
                          else subset.groupby(key))
                for aggregate in ['count', 'mean_pct']:
                    name = f'auto__{entity}__{days}d__{suffix}__{aggregate}'
                    values = groups.size() if aggregate == 'count' else groups['watched_pct'].mean()
                    if key is None:
                        out[name] = map_genres(values, count=aggregate == 'count')
                    else:
                        out[name] = out[key].map(values)
                        if aggregate == 'count':
                            out[name] = out[name].fillna(0)
                    auto_columns.append(name)
    unknown_genres = ~out['item_id'].isin(genres['item_id'])
    genre_features = [c for c in [*MANUAL_FEATURES,*auto_columns]
                      if c.startswith('ui_genre_') or c.startswith('auto__user_genre__')]
    out.loc[unknown_genres,genre_features] = np.nan
    unknown_type = out['item_id'].map(content).isna()
    out.loc[unknown_type,['ui_type_share','ui_type_share7','ui_type_share30']] = np.nan
    for c in [*COMMON_FEATURES, *MANUAL_FEATURES, *auto_columns]:
        out[c] = out[c].astype('float32')
    numeric = out[[*COMMON_FEATURES, *MANUAL_FEATURES, *auto_columns]].to_numpy()
    if np.isinf(numeric).any():
        raise ValueError('Infinite feature value')
    return out, auto_columns, {'manual_feature_seconds': manual_seconds,
                               'auto_feature_seconds': perf_counter()-auto_start}


def deduplicate_features(frame, columns, protected=()):
    """Fit a constant/exact-duplicate filter only on training feature values."""
    kept = list(protected)
    dropped = {}
    for column in columns:
        if column in kept:
            continue
        if frame[column].nunique(dropna=False) <= 1:
            dropped[column] = 'constant'
            continue
        duplicate = next((c for c in kept if frame[column].equals(frame[c])), None)
        if duplicate is not None:
            dropped[column] = f'duplicate of {duplicate}'
        else:
            kept.append(column)
    return [c for c in kept if c not in protected], dropped
