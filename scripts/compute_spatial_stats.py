#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the spatial normalisation statistics from the training storms.

The released ``data/spatial_stats_train2003_2022.pkl`` was produced by this
script and is what ``run_single_track.py`` loads by default.

**This script cannot be run from the public release.** It needs the 2003-2022
training storms, which are not distributed; only the 2024 North Atlantic test
season is. It is published as the provenance of the statistics file: the
statistics are part of the model definition, change them and the predictions
change, so the exact procedure that produced them is stated here in code rather
than left to be taken on trust. The file it produces is under 1 KB and is
committed directly, so nothing in the reproduction path depends on running it.

Statistics are a per-variable, per-level mean and standard deviation over the
first ``--max-samples`` training storms, in the order
:func:`fastml.data.find_storm_dirs` yields them. Values are accumulated in a
streaming pass rather than by concatenating the storms, which would need about
10 GB of memory.

Usage::

    python scripts/compute_spatial_stats.py \
        --data-dir data/training_data \
        --out data/spatial_stats_train2003_2022.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastml.config import (  # noqa: E402
    DATA_DIR,
    N_2D_VARS,
    N_3D_VARS,
    N_LEVELS,
    TARGET_SEQ_LEN,
    TRAINING_DATA_DIR,
)
from fastml.data import find_storm_dirs, load_spatial, load_storm  # noqa: E402

#: Training years, and the duration filter in force when the published
#: statistics were generated.
TRAIN_YEARS = tuple(range(2003, 2023))
TRAIN_MIN_DURATION_H = 72


class _StreamingMoments:
    """Accumulate count, sum and sum of squares over finite values."""

    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=np.int64)
        self.total = np.zeros(shape, dtype=np.float64)
        self.total_sq = np.zeros(shape, dtype=np.float64)

    def update(self, index, values):
        finite = values[np.isfinite(values)].astype(np.float64)
        self.count[index] += finite.size
        self.total[index] += finite.sum()
        self.total_sq[index] += np.square(finite).sum()

    def mean_std(self, index):
        n = self.count[index]
        if n == 0:
            return 0.0, 1e-8
        mean = self.total[index] / n
        # Population variance, matching numpy's default ddof=0.
        var = max(self.total_sq[index] / n - mean * mean, 0.0)
        return float(mean), max(float(np.sqrt(var)), 1e-8)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute spatial normalisation statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--out", type=Path,
                        default=DATA_DIR / "spatial_stats_train2003_2022.pkl")
    parser.add_argument("--max-samples", type=int, default=30,
                        help="Number of training storms to draw statistics from")
    parser.add_argument("--seq-len", type=int, default=TARGET_SEQ_LEN)
    args = parser.parse_args()

    storm_dirs = find_storm_dirs(args.data_dir, years=TRAIN_YEARS)
    print(f"Training-year storm directories: {len(storm_dirs)}")

    moments_3d = _StreamingMoments((N_3D_VARS, N_LEVELS))
    moments_2d = _StreamingMoments((N_2D_VARS,))

    used = 0
    for storm_dir in storm_dirs:
        if used >= args.max_samples:
            break
        storm, reason = load_storm(
            storm_dir, seq_len=args.seq_len, min_duration_h=TRAIN_MIN_DURATION_H
        )
        if storm is None:
            continue
        x3d, x2d = load_spatial(storm, seq_len=args.seq_len)
        for vi in range(N_3D_VARS):
            for li in range(N_LEVELS):
                moments_3d.update((vi, li), x3d[:, :, vi, li].ravel())
        for vi in range(N_2D_VARS):
            moments_2d.update((vi,), x2d[:, :, vi].ravel())
        used += 1
        print(f"  [{used}/{args.max_samples}] {storm['storm_id']}")

    if used == 0:
        print("No training storms available; cannot compute statistics.")
        return 1

    stats = {
        "3d": [[moments_3d.mean_std((vi, li)) for li in range(N_LEVELS)]
               for vi in range(N_3D_VARS)],
        "2d": [moments_2d.mean_std((vi,)) for vi in range(N_2D_VARS)],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(stats, f)

    names_3d = ["T", "Q", "U", "V", "Z"]
    print(f"\nStatistics from {used} storms -> {args.out}")
    for vi, name in enumerate(names_3d):
        head = "  ".join(f"{mu:>10.3f}/{sd:<9.3f}" for mu, sd in stats["3d"][vi])
        print(f"  {name}: {head}")
    for vi, name in enumerate(["SST", "MSLP"]):
        mu, sd = stats["2d"][vi]
        print(f"  {name}: {mu:.3f}/{sd:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
