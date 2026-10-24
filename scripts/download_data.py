#!/usr/bin/env python3
"""Download the full KION archive (~79 MB) at a pinned commit into data/raw and verify its SHA-256."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recsys.common.data import download  # noqa: E402

if __name__ == '__main__':
    print(json.dumps(download(json.loads((ROOT / 'config.json').read_text())), indent=2))
