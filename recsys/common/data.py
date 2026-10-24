"""Pinned KION download, temporal partitions, targets and user blocks."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/data_original.zip'


def log(message):
    print(f'[{datetime.now().astimezone().strftime("%H:%M:%S")}] {message}', flush=True)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def download(config):
    """Fetch the complete three-table archive at a pinned commit and check its checksum.

    The separate interactions.csv.zip in the source repository is truncated, so the full
    data_original.zip is the authoritative input.
    """
    url = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{config['dataset_commit']}/data_original.zip"
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        part = RAW.with_suffix('.zip.part')
        subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '15',
                        '--max-time', '600', url, '-o', str(part)], check=True)
        part.replace(RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != config['dataset_archive_sha256']:
        raise ValueError(f'KION archive checksum {digest} does not match the pinned input')
    return {'url': url, 'bytes': RAW.stat().st_size, 'sha256': digest}


def load_raw(config):
    manifest = download(config)
    cache = ROOT / 'data/cache'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"interactions_{manifest['sha256'][:12]}.parquet"
    if path.exists():
        events = pd.read_parquet(path)
    else:
        with zipfile.ZipFile(RAW) as archive, archive.open('data_original/interactions.csv') as stream:
            events = pd.read_csv(stream)
        events['last_watch_dt'] = pd.to_datetime(events['last_watch_dt'], format='%Y-%m-%d')
        events['user_id'] = events['user_id'].astype('int32')
        events['item_id'] = events['item_id'].astype('int32')
        events.to_parquet(path, index=False)
    if len(events) != 5_476_251:
        raise ValueError(f'Incomplete KION input: {len(events)} rows, expected 5,476,251')
    with zipfile.ZipFile(RAW) as archive:
        with archive.open('data_original/users.csv') as stream:
            users = pd.read_csv(stream)
        with archive.open('data_original/items.csv') as stream:
            items = pd.read_csv(stream)
    return events, users, items


def split_dates(events):
    # A complete calendar week is Monday 00:00 through next Monday 00:00.
    observed_end = events['last_watch_dt'].max().normalize() + pd.Timedelta(days=1)
    end = observed_end - pd.Timedelta(days=observed_end.weekday())
    return {name: (end - pd.Timedelta(days=offset), end - pd.Timedelta(days=offset-7))
            for name, offset in [('train', 21), ('validation', 14), ('test', 7)]}


def target_map(events, cutoff, end):
    target = events[(events['last_watch_dt'] >= cutoff) & (events['last_watch_dt'] < end)]
    warm = set(events.loc[events['last_watch_dt'] < cutoff, 'user_id'])
    return {int(u): set(map(int, g['item_id'])) for u, g in target.groupby('user_id', sort=True)
            if u in warm}


def user_blocks(events, cutoff, excluded, size, n_blocks, seed):
    """Seeded permutation of users with history before cutoff, cut into equal blocks."""
    eligible = np.sort(events.loc[events.last_watch_dt < cutoff, 'user_id'].unique())
    eligible = eligible[~np.isin(eligible, excluded)]
    order = np.random.default_rng(seed).permutation(eligible)
    return [np.sort(order[k * size:(k + 1) * size]) for k in range(n_blocks)]
