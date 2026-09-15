#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load per-storm ERA5 inputs and assemble model-ready tensors.

Each storm directory under ``training_data/<year>/<storm_id>/`` holds:

``<storm_id>_dataset.pkl``
    Track, observed intensity, FAST scalars (alpha, beta, gamma, vp), the ERA5
    reference ventilation terms ``chi_ref``/``s_ref``, environmental winds and
    translation velocity. This is what the FAST baseline needs.

``<storm_id>_spatial_1000km.pkl``
    The 72x72 storm-centred fields (T, Q, U, V, Z on 7 levels, plus SST and
    MSLP) that are the actual input to the CNN. Roughly 100 MB per storm, so
    these are distributed separately -- see ``scripts/download_data.py``.

Time axis convention, matched to :func:`fastml.fast_physics.run_fast_with_init`::

    t=0                      t_start (=48)                    seq_valid
     |<- 48 h initialisation ->|<--------- forecast --------->|

Sequences are cut at ``t0 = max(0, t_45 - 48)`` so that exactly 48 h of history
precede the forecast start, then edge-padded to a fixed length.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import (
    MIN_DURATION_H,
    MIN_VMAX_KTS,
    MS_TO_KNOTS,
    N_2D_VARS,
    N_3D_VARS,
    N_LEVELS,
    SPATIAL_H,
    SPATIAL_W,
    TARGET_SEQ_LEN,
    VMAX_START_KTS,
)

#: Scalar/vector fields carried through trimming and padding, with their width.
_SERIES_KEYS = {
    "scalars": 4,
    "chi_ref": 1,
    "s_ref": 1,
    "xs_ref": 1,
    "v_init": 1,
    "v_gt": 1,
    "env_wnds": 4,
    "utran": 1,
    "vtran": 1,
}


def _trim_axis1(arr, t0):
    """Drop the first ``t0`` timesteps, whether the array is 1-D or ``[1, T, ...]``."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[t0:]
    sl = [slice(None)] * arr.ndim
    sl[1] = slice(t0, None)
    return arr[tuple(sl)]


def _pad_axis1(arr, tlen, axis=1):
    """Edge-pad (or truncate) to ``tlen`` along the time axis.

    Padding repeats the final timestep. Padded steps sit beyond ``seq_valid``
    and are masked out of every metric, so their value only has to be finite.
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        n = arr.shape[0]
        if n >= tlen:
            return arr[:tlen]
        return np.concatenate([arr, np.repeat(arr[-1:], tlen - n)])
    if arr.ndim <= axis:
        return arr
    n = arr.shape[axis]
    if n >= tlen:
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(0, tlen)
        return arr[tuple(sl)]
    last = [slice(None)] * arr.ndim
    last[axis] = slice(n - 1, n)
    return np.concatenate([arr, np.repeat(arr[tuple(last)], tlen - n, axis=axis)], axis=axis)


def _pad_times(times, n_valid, tlen):
    """Extend an hourly timestamp array to ``tlen`` entries."""
    import pandas as pd

    if times is None:
        return None
    ta = np.asarray(times).ravel()[:n_valid]
    if len(ta) < tlen:
        extra = pd.date_range(
            start=pd.Timestamp(ta[-1]) + pd.Timedelta(hours=1),
            periods=tlen - len(ta),
            freq="h",
        )
        ta = np.concatenate([ta, extra.to_numpy()])
    return ta[:tlen]


def find_storm_dirs(data_dir, years=None, basin=None, storm_names=None):
    """List storm directories, optionally filtered by year, basin, or name.

    Args:
        data_dir: Root containing ``<year>/<storm_id>/`` subdirectories.
        years: Iterable of years to keep; ``None`` keeps all.
        basin: ``'AL'``/``'NA'`` for North Atlantic or ``'EP'`` for East Pacific,
            matched against the storm-id prefix.
        storm_names: Case-insensitive substrings to match against the storm id.

    Returns:
        Sorted list of directories that contain a ``*_dataset.pkl``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run scripts/download_data.py to fetch the per-storm ERA5 inputs."
        )

    prefix = {"NA": "AL", "AL": "AL", "EP": "EP"}.get((basin or "").upper())
    years = set(int(y) for y in years) if years else None
    names = [n.upper() for n in storm_names] if storm_names else None

    out = []
    for year_dir in sorted(data_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        if years is not None and int(year_dir.name) not in years:
            continue
        for storm_dir in sorted(year_dir.iterdir()):
            if not storm_dir.is_dir():
                continue
            if prefix and not storm_dir.name.upper().startswith(prefix):
                continue
            if names and not any(n in storm_dir.name.upper() for n in names):
                continue
            if not list(storm_dir.glob("*_dataset.pkl")):
                continue
            out.append(storm_dir)
    return out


def load_storm(storm_dir, seq_len=TARGET_SEQ_LEN,
               min_vmax_kts=MIN_VMAX_KTS, min_duration_h=MIN_DURATION_H,
               require_spatial=True):
    """Load and time-align one storm.

    Args:
        storm_dir: Directory holding the storm's pickles.
        seq_len: Padded sequence length.
        min_vmax_kts: Reject storms that never reach this peak intensity, since
            the 45 kt forecast start would never trigger.
        min_duration_h: Reject storms with fewer than this many hours after the
            45 kt threshold, which are too short to score a forecast against.
        require_spatial: Reject storms without the 72x72 fields the CNN needs.

    Returns:
        A dict of padded arrays, or ``None`` if the storm fails a filter. The
        ``skip_reason`` of a rejected storm is reported via :func:`load_storms`.
    """
    storm_dir = Path(storm_dir)
    dataset_pkl = sorted(storm_dir.glob("*_dataset.pkl"))
    if not dataset_pkl:
        return None, "no *_dataset.pkl"
    with open(dataset_pkl[0], "rb") as f:
        data = pickle.load(f)

    spatial_pkl = sorted(storm_dir.glob("*_spatial_1000km.pkl"))
    if require_spatial and not spatial_pkl:
        return None, "missing *_spatial_1000km.pkl (run scripts/download_data.py)"

    v_gt_full = np.asarray(data["v_gt"])[0, :, 0]
    kts = v_gt_full * MS_TO_KNOTS
    # Threshold in m/s, the unit the data is stored in, so that a storm peaking
    # exactly at 45 kt is not decided by a rounding artefact of the conversion.
    peak_ms = float(np.nanmax(v_gt_full)) if np.any(np.isfinite(v_gt_full)) else 0.0
    if peak_ms < min_vmax_kts / MS_TO_KNOTS:
        return None, f"peak {peak_ms * MS_TO_KNOTS:.1f} kt < {min_vmax_kts:.0f} kt"

    t_45 = next(
        (t for t in range(len(kts)) if np.isfinite(kts[t]) and kts[t] >= VMAX_START_KTS),
        None,
    )
    if t_45 is None:
        return None, f"never reaches {VMAX_START_KTS:.0f} kt"

    forecast_h = len(kts) - t_45
    if forecast_h < min_duration_h:
        return None, f"only {forecast_h} h after {VMAX_START_KTS:.0f} kt (< {min_duration_h})"

    # 48 h of history before the forecast start.
    t0 = max(0, t_45 - 48)
    n_valid_full = len(kts) - t0
    seq_valid = min(n_valid_full, seq_len)

    storm = {
        "hurricane": data.get("hurricane", storm_dir.name),
        "storm_id": storm_dir.name,
        "year": int(storm_dir.parent.name),
        "seq_valid": int(seq_valid),
        "t0": int(t0),
        "spatial_path": str(spatial_pkl[0]) if spatial_pkl else None,
    }
    for key, width in _SERIES_KEYS.items():
        arr = data.get(key)
        if arr is None:
            storm[key] = np.zeros((1, seq_len, width), np.float32)
        else:
            storm[key] = _pad_axis1(_trim_axis1(arr, t0), seq_len).astype(np.float32)
    for key in ("lats", "lons"):
        arr = data.get(key)
        if arr is None:
            storm[key] = np.zeros((1, seq_len), np.float32)
        else:
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            storm[key] = _pad_axis1(_trim_axis1(arr, t0), seq_len).astype(np.float64)
    storm["times"] = _pad_times(
        np.asarray(data["times"]).ravel()[t0:] if data.get("times") is not None else None,
        n_valid_full,
        seq_len,
    )
    return storm, None


def load_storms(storm_dirs, verbose=True, **kwargs):
    """Load several storms, reporting which were skipped and why."""
    storms, skipped = [], []
    for storm_dir in storm_dirs:
        storm, reason = load_storm(storm_dir, **kwargs)
        if storm is None:
            skipped.append((Path(storm_dir).name, reason))
            if verbose:
                print(f"  skip {Path(storm_dir).name}: {reason}")
        else:
            storms.append(storm)
    if verbose:
        print(f"Loaded {len(storms)} storms ({len(skipped)} skipped)")
    return storms, skipped


def load_spatial(storm, seq_len=TARGET_SEQ_LEN):
    """Read the 72x72 fields for one storm, trimmed and padded to match.

    Loaded on demand rather than held in memory: the full archive is tens of
    gigabytes, while one storm at a time is about 100 MB.

    Returns:
        ``(x3d, x2d)`` shaped ``[1, seq_len, 5, 7, 72, 72]`` and
        ``[1, seq_len, 2, 72, 72]``.
    """
    path = storm.get("spatial_path")
    t0 = storm.get("t0", 0)
    if path is None:
        return (
            np.zeros((1, seq_len, N_3D_VARS, N_LEVELS, SPATIAL_H, SPATIAL_W), np.float32),
            np.zeros((1, seq_len, N_2D_VARS, SPATIAL_H, SPATIAL_W), np.float32),
        )
    with open(path, "rb") as f:
        sp = pickle.load(f)
    x3d = sp.get("spatial_3d_1km")
    x2d = sp.get("spatial_2d_1km")
    if x3d is None or x2d is None:
        raise KeyError(f"{path} lacks spatial_3d_1km / spatial_2d_1km")
    x3d = _pad_axis1(x3d[:, t0:].astype(np.float32), seq_len)
    x2d = _pad_axis1(x2d[:, t0:].astype(np.float32), seq_len)
    return x3d, x2d


def load_spatial_stats(path):
    """Load the per-variable, per-level normalisation statistics.

    The released statistics come from the 2003-2021 training storms. Reusing
    them -- rather than recomputing from whatever storms happen to be present --
    is what makes a single-storm run reproduce the published numbers exactly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalisation statistics not found: {path}")
    with open(path, "rb") as f:
        stats = pickle.load(f)
    if "3d" not in stats or "2d" not in stats:
        raise KeyError(f"{path} is not a spatial-stats pickle (needs '3d' and '2d')")
    return stats


def normalize_spatial(x3d, x2d, stats):
    """Z-score each variable and level independently.

    Per-level normalisation is deliberate: a single statistic per variable would
    be dominated by the vertical mean profile and would wash out the vertical
    gradients that the ventilation diagnosis depends on.
    """
    x3d = np.array(x3d, dtype=np.float32, copy=True)
    x2d = np.array(x2d, dtype=np.float32, copy=True)
    for vi, levels in enumerate(stats["3d"]):
        for li, (mu, std) in enumerate(levels):
            x3d[:, :, vi, li] = np.nan_to_num(
                (x3d[:, :, vi, li] - mu) / std, nan=0.0, posinf=0.0, neginf=0.0
            )
    for vi, (mu, std) in enumerate(stats["2d"]):
        x2d[:, :, vi] = np.nan_to_num(
            (x2d[:, :, vi] - mu) / std, nan=0.0, posinf=0.0, neginf=0.0
        )
    return x3d, x2d
