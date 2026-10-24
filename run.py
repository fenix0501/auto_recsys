#!/usr/bin/env python3
"""Run the unified study: manual FE and generators V1-V4 on one split.

    python run.py                                  # everything, as in the report
    python run.py --pipelines manual               # manual pipeline only
    python run.py --pipelines v2 v4 --output results/auto
"""
import argparse
from pathlib import Path

from recsys.study import PIPELINES, run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pipelines', nargs='+', choices=PIPELINES, default=PIPELINES)
    parser.add_argument('--output', default='results/unified')
    parser.add_argument('--block-size', type=int, default=20000, help='users per block (train = 1 block, val/test = 2)')
    parser.add_argument('--seed', type=int, default=2026, help='user permutation and bootstrap seed')
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    parser.add_argument('--seeds', type=int, default=5, help='bagging seeds of every final model')
    parser.add_argument('--max-trees', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--n-jobs', type=int, default=None)
    args = parser.parse_args()
    run(ROOT / args.output, args.block_size, args.seed, args.max_trees, args.patience,
        args.bootstrap_repeats, args.seeds, args.n_jobs, args.pipelines)


if __name__ == '__main__':
    main()
