#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-track inference: run FAST and FAST_ML over the same storm.

The comparison is constructed so that the ventilation index is the only
difference between the two forecasts:

===============  ==========================================================
FAST             ``chi * S`` from ERA5 diagnostics, with ``chi_ref`` passed
                 through the same mean-to-90th-percentile calibration.
FAST_ML          ``chi * S`` from the two-stream CNN.
===============  ==========================================================

Both then go through the identical NumPy ODE with identical scalars, track,
environmental winds, drag field, 48 h initialisation and forcing decay. Any
skill difference is therefore attributable to the ventilation diagnosis.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import MS_TO_KNOTS, XS_NAN_FALLBACK
from .data import load_spatial, normalize_spatial
from .fast_physics import (
    chi_calibrated_multiply,
    find_t_start,
    median_filter_1d,
    run_fast_with_init,
)


def _masked_scores(pred_ms, obs_ms, mask):
    """RMSE (kt), bias (kt) and correlation over a validity mask."""
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    d = pred_ms[mask] - obs_ms[mask]
    rmse = float(np.sqrt(np.mean(d ** 2))) * MS_TO_KNOTS
    bias = float(np.mean(d)) * MS_TO_KNOTS
    p, o = pred_ms[mask], obs_ms[mask]
    if mask.sum() > 1 and np.std(p) > 1e-12 and np.std(o) > 1e-12:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = float(np.corrcoef(p, o)[0, 1])
        corr = corr if np.isfinite(corr) else np.nan
    else:
        corr = np.nan
    return rmse, bias, corr


def predict_chi_s(model, storm, stats, device="cpu"):
    """Run the CNN over one storm to obtain the ML ventilation terms.

    Returns:
        ``(chi, s)`` each ``[T]``, with chi already calibrated onto [0, 4].
    """
    x3d, x2d = load_spatial(storm, seq_len=storm["scalars"].shape[1])
    x3d, x2d = normalize_spatial(x3d, x2d, stats)

    # The model sees the same median-filtered vp that the ODE integrates, so the
    # two paths cannot disagree about the potential intensity.
    sc = np.array(storm["scalars"], dtype=np.float64, copy=True)
    sc[0, :, 3] = median_filter_1d(sc[0, :, 3], size=3)

    with torch.no_grad():
        out = model(
            torch.from_numpy(x3d).float().to(device),
            torch.from_numpy(x2d).float().to(device),
            torch.from_numpy(sc).float().to(device),
        )
    chi = out["chi"][0, :, 0].cpu().numpy().astype(np.float64)
    s = out["s"][0, :, 0].cpu().numpy().astype(np.float64)
    return chi, s


def run_storm(model, storm, stats, device="cpu", vent_scale=1.0):
    """Produce the FAST and FAST_ML intensity forecasts for one storm.

    Args:
        model: Loaded :class:`fastml.model.TwoStreamFASTModel`.
        storm: A storm dict from :func:`fastml.data.load_storm`.
        stats: Normalisation statistics from :func:`fastml.data.load_spatial_stats`.
        device: Torch device for the CNN.
        vent_scale: Diagnostic multiplier on the ML ventilation index; 1.0
            reproduces the published configuration.

    Returns:
        Dict of ``[T]`` time series in knots (``*_vmax_kts``, ``v_obz_kts``,
        ``vp_kts``), the ventilation terms, the track, and scalar metrics.
    """
    T = storm["scalars"].shape[1]
    seq_valid = storm["seq_valid"]

    # ── Shared ODE inputs ────────────────────────────────────────────────────
    scalars = np.array(storm["scalars"], dtype=np.float64, copy=True)
    scalars[0, :, 3] = median_filter_1d(scalars[0, :, 3], size=3)

    v_gt = np.asarray(storm["v_gt"], dtype=np.float64)
    env_wnds = np.asarray(storm["env_wnds"], dtype=np.float64)
    utran = np.asarray(storm["utran"], dtype=np.float64)
    vtran = np.asarray(storm["vtran"], dtype=np.float64)
    lats = np.asarray(storm["lats"], dtype=np.float64)
    lons = np.asarray(storm["lons"], dtype=np.float64)

    # ── FAST_ML path: ventilation from the CNN ───────────────────────────────
    ml_chi, ml_s = predict_chi_s(model, storm, stats, device=device)
    xs_ml = np.maximum(
        np.nan_to_num(ml_chi * ml_s * vent_scale, nan=XS_NAN_FALLBACK),
        XS_NAN_FALLBACK,
    ).reshape(1, T, 1)
    ml_v, ml_vmax, ml_m = run_fast_with_init(
        scalars, xs_ml, v_gt, env_wnds, utran, vtran, lats,
        ml_s.reshape(1, T), lons=lons,
    )

    # ── FAST path: ventilation from ERA5 ─────────────────────────────────────
    chi_ref = np.asarray(storm["chi_ref"], dtype=np.float64)
    s_ref = np.asarray(storm["s_ref"], dtype=np.float64)
    fast_chi = chi_calibrated_multiply(chi_ref)
    xs_ref = np.maximum(
        np.nan_to_num(fast_chi * s_ref, nan=XS_NAN_FALLBACK),
        XS_NAN_FALLBACK,
    )
    fast_s = s_ref[0, :, 0]
    fast_v, fast_vmax, fast_m = run_fast_with_init(
        scalars, xs_ref, v_gt, env_wnds, utran, vtran, lats,
        fast_s.reshape(1, T), lons=lons,
    )

    # ── Scoring ──────────────────────────────────────────────────────────────
    v_obz = v_gt[0, :, 0]
    base_mask = (np.arange(T) < seq_valid) & np.isfinite(v_obz) & (v_obz > 0)
    ml_mask = base_mask & np.isfinite(ml_vmax) & (ml_vmax > 0)
    fast_mask = base_mask & np.isfinite(fast_vmax) & (fast_vmax > 0)

    ml_rmse, ml_bias, ml_corr = _masked_scores(ml_vmax, v_obz, ml_mask)
    fast_rmse, fast_bias, fast_corr = _masked_scores(fast_vmax, v_obz, fast_mask)
    t_start, _ = find_t_start(v_obz, T)

    return {
        "storm_id": storm["storm_id"],
        "hurricane": storm["hurricane"],
        "year": storm["year"],
        "times": storm["times"],
        "seq_valid": seq_valid,
        "t_start": int(t_start),
        "T": T,
        # Intensity time series, knots.
        "v_obz_kts": v_obz * MS_TO_KNOTS,
        "fast_vmax_kts": fast_vmax * MS_TO_KNOTS,
        "ml_vmax_kts": ml_vmax * MS_TO_KNOTS,
        "fast_v_axisym_kts": fast_v * MS_TO_KNOTS,
        "ml_v_axisym_kts": ml_v * MS_TO_KNOTS,
        "vp_kts": scalars[0, :, 3] * MS_TO_KNOTS,
        # Ventilation diagnosis.
        "fast_chi": fast_chi.reshape(-1)[:T],
        "fast_s": fast_s,
        "fast_vent": fast_chi.reshape(-1)[:T] * fast_s,
        "ml_chi": ml_chi,
        "ml_s": ml_s,
        "ml_vent": ml_chi * ml_s,
        "fast_m": fast_m,
        "ml_m": ml_m,
        # Track.
        "lats": np.asarray(lats).reshape(-1)[:T],
        "lons": np.asarray(lons).reshape(-1)[:T],
        # Metrics.
        "fast_rmse_kts": fast_rmse,
        "fast_bias_kts": fast_bias,
        "fast_corr": fast_corr,
        "ml_rmse_kts": ml_rmse,
        "ml_bias_kts": ml_bias,
        "ml_corr": ml_corr,
        "rmse_gain_kts": (
            fast_rmse - ml_rmse
            if np.isfinite(fast_rmse) and np.isfinite(ml_rmse)
            else np.nan
        ),
        "n_valid": int(base_mask.sum()),
        "peak_obs_kts": float(np.nanmax(v_obz[:seq_valid])) * MS_TO_KNOTS,
        "peak_fast_kts": float(np.nanmax(fast_vmax[:seq_valid])) * MS_TO_KNOTS,
        "peak_ml_kts": float(np.nanmax(ml_vmax[:seq_valid])) * MS_TO_KNOTS,
    }
