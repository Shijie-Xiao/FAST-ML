#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wind-speed exceedance probability maps for the FAST / FAST-ML ensembles.

Applies the hazard-map methodology of ``Reproduce/strike_probability.py``
(Lin et al. 2020 Fig. 2 style wind speed probability) to the two GEFS-driven
ODE ensembles in ``data/ensemble/`` (Flossie, Priscilla).

Wind model - corrected to match the paper exactly
--------------------------------------------------
Lin et al. (2023), Appendix A (Eqs. A1-A2), identical to the official
implementation ``wind/tc_wind.py`` of github.com/linjonathan/tropical_cyclone_risk:

    v_net = v_axi + G(phi)*u_t + 0.1 * S * (v / 15)        (A1)
    G(phi) = min[1, 0.8 + 0.35*(1 + tanh((phi-35)/10))]    (A2)

with the official cap that the wind increment magnitude may not exceed 50% of
the axisymmetric wind, and centred-difference translation speeds
(``util/sphere.calc_translational_speed``).

Differences w.r.t. ``Reproduce/wind_field.py`` (which were checked against the
paper and the official repository):

1. SHEAR TERM UNITS - the reference converts v to knots before ``0.1*S*(v/15)``;
   the paper and official code use v in m/s.  Fixed here (factor 1.94).
2. 50% INCREMENT CAP - ``mag_fac = min(1, 0.5*v/|inc|)`` from the official
   ``tc_wind.py`` was missing in the reference.  Added here, applied locally
   (increment at radius r capped by 0.5*V_sym(r)).
3. TRANSLATION SPEED - centred differences with edge extrapolation, exactly as
   ``util/sphere.py``; the reference used backward differences.

The radial structure needed to map the storm onto a grid uses the Holland
profile of ``Reproduce/wind_field.py`` (B=1.5, r_m=40 km, r_0=700 km), i.e.
V_sym(r) replaces the scalar v in the asymmetric part, as in
``strike_probability.py``.

Probability definition (Lin et al. 2020 Fig. 2):
    P(kt, x) = (# ensemble members whose modelled wind at grid point x
                 reaches `kt` at any sampled time) / n_members

Note ``strike_probability.py`` divides by the per-point sampling count
(``total_count``); that inflates probabilities at swath edges.  The
member-fraction definition above is the standard strike-probability
definition used in the paper's hazard maps and is what is plotted here.

Usage:
    python scripts/plot_strike_probability.py                      # both storms, FAST-ML
    python scripts/plot_strike_probability.py --case flossie
    python scripts/plot_strike_probability.py --model fast         # FAST instead
    python scripts/plot_strike_probability.py --model both         # + comparison fig
    python scripts/plot_strike_probability.py --verify             # self-check & exit

Outputs to ``results/ensemble/``:
    strike_probability_<case>_<model>.png/.svg   probability panels
    tracks_<case>_<model>.png                     ensemble tracks
    strike_probability_<case>.nc                  probability fields
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:                       # normal import inside the repo
    from fastml.config import ENSEMBLE_DIR, RESULTS_DIR  # noqa: E402
except Exception:          # plotting-only environments without torch
    ENSEMBLE_DIR = REPO_ROOT / "data" / "ensemble"
    RESULTS_DIR = REPO_ROOT / "results"

KT_TO_MS = 0.514444
EARTH_R = 6378100.0            # m - matches official util/constants.py & Reproduce
HOLLAND_B = 1.5                # Reproduce/wind_field.py ChavasWindModel default

# Wind-model parameters (strike_probability.py defaults)
R_M = 40_000.0                 # radius of maximum wind (m)
R_0 = 700_000.0                # outer radius (m)
K_SHAPE = 1.0                  # outer shape parameter
U_SHEAR, V_SHEAR = 5.0, 0.0    # default shear (m/s) - no ERA5 for the 2025 cases
THRESHOLDS_KT = [34, 50, 64, 83, 96]
GRID_RES = 0.25                # degrees (wind-speed probability grid)
TIME_SAMPLING_HOURS = 6.0
MAX_DISTANCE_KM = 400.0        # strike_probability.py default

# ── Track strike probability: Reproduce/track_model/visualize_strike_prob_75km.py ──
STRIKE_RADIUS_KM = 75.0        # radius_km of the reference script
STRIKE_GRID_RES = 0.2          # grid_res used by run_visualize_strike_prob
STRIKE_PAD_DEG = 1.0           # grid extends 1 deg beyond the track bounds
STRIKE_MAX_DAYS = 5.0          # max_days window of the reference script

CASES = {
    "flossie": {
        "ode_nc": "flossie/ode_flossie_midfinit_finit.nc",
        "storm": "Flossie",
        "init_time": "2025-06-29 12:00 UTC",
        "lon_pad": 7.0,
        "lat_pad": 7.0,
    },
    "priscilla": {
        "ode_nc": "priscilla/ode_priscilla_vp1p10_free.nc",
        "storm": "Priscilla",
        "init_time": "2025-10-04 18:00 UTC",
        "lon_pad": 7.0,
        "lat_pad": 7.0,
    },
}

MODEL_KEYS = {"fast": "fast_vmax_kts", "ml": "ml_vmax_kts"}
MODEL_LABELS = {"fast": "FAST", "ml": "FAST-ML"}

#: Raw (pre-sampling) GEFS 31-member tracks, copied into the repo for
#: reproducibility; drawn by the reference visualize_strike_prob_75km.py via
#: its lon_orig/lat_orig argument.
RAW_TRACKS = {
    "flossie": ENSEMBLE_DIR / "flossie" / "gefs_raw_31members_20250629T120000.pkl",
    "priscilla": ENSEMBLE_DIR / "priscilla" / "gefs_raw_31members_20251004T180000.pkl",
}


def load_raw_gefs_tracks(pkl_path, max_days=None):
    """Load the raw (pre-sampling) 31-member GEFS track ensemble.

    Mirrors the lon_orig/lat_orig handling of
    Reproduce/track_model/visualize_strike_prob_75km.py: reads the raw pickle,
    selects the gefs/kwbc members, applies the optional time window, and
    pads members of unequal length with NaN.
    """
    import pickle
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    tracks = [t for t in raw.get("tracks", [])
              if t.get("ensemble_system") in ("gefs", "kwbc")]
    if not tracks:                       # fall back to whatever is there
        tracks = raw.get("tracks", [])
    lon_list, lat_list = [], []
    for tr in tracks:
        lon_arr, lat_arr = np.asarray(tr["lon"], float), np.asarray(tr["lat"], float)
        dt = tr.get("datetime")
        if dt is not None and max_days is not None and len(dt) == len(lon_arr):
            t0 = dt[0]
            m = np.array([(t - t0).total_seconds() <= max_days * 86400 + 1e-6
                          for t in dt])
            lon_arr, lat_arr = lon_arr[m], lat_arr[m]
        lon_list.append(lon_arr)
        lat_list.append(lat_arr)
    if not lon_list:
        return None, None
    max_len = max(len(x) for x in lon_list)
    lon_o = np.full((len(lon_list), max_len), np.nan)
    lat_o = np.full((len(lat_list), max_len), np.nan)
    for i, (lo, la) in enumerate(zip(lon_list, lat_list)):
        lon_o[i, :len(lo)], lat_o[i, :len(la)] = lo, la
    return lon_o, lat_o


# ═════════════════════════════════════════════════════════════════════════════
# Wind model - vectorised, matches the paper / official tc_wind.py
# ═════════════════════════════════════════════════════════════════════════════

def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km (same formula as official util/sphere.py)."""
    lon1r, lat1r = np.deg2rad(np.asarray(lon1, float)), np.deg2rad(np.asarray(lat1, float))
    lon2r, lat2r = np.deg2rad(np.asarray(lon2, float)), np.deg2rad(np.asarray(lat2, float))
    a = (np.sin((lat2r - lat1r) / 2) ** 2
         + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2)
    return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) / 1000.0


def wrap180(lon):
    lon = np.asarray(lon, float)
    return np.where(lon > 180, lon - 360, np.where(lon < -180, lon + 360, lon))


def translation_velocity(lon, lat, time_h):
    """Centred-difference (u, v) translation speed in m/s.

    Mirrors ``util/sphere.calc_translational_speed``: points are extrapolated
    at the edges (2*x0 - x1), the displacement magnitude comes from a
    haversine distance evaluated at the centre latitude/longitude with the
    sign of the coordinate increment, halved, and divided by the (seconds)
    centre time step.
    """
    lon = wrap180(lon)
    e_lon = np.concatenate(([2 * lon[0] - lon[1]], lon, [2 * lon[-1] - lon[-2]]))
    e_lat = np.concatenate(([2 * lat[0] - lat[1]], lat, [2 * lat[-1] - lat[-2]]))
    e_t = np.concatenate(([2 * time_h[0] - time_h[1]], time_h,
                          [2 * time_h[-1] - time_h[-2]]))

    # signed half-displacements (km) via haversine at the centre point
    dlon_sign = np.sign(e_lon[2:] - e_lon[:-2])
    dlon_sign[dlon_sign == 0] = 1.0
    dx_km = 0.5 * dlon_sign * haversine_km(e_lon[2:], lat, e_lon[:-2], lat)
    dlat_sign = np.sign(e_lat[2:] - e_lat[:-2])
    dlat_sign[dlat_sign == 0] = 1.0
    dy_km = 0.5 * dlat_sign * haversine_km(lon, e_lat[2:], lon, e_lat[:-2])

    dt_s = (e_t[2:] - e_t[:-2]) * 3600.0
    dt_safe = np.where(dt_s > 0, dt_s, 1.0)
    u = np.where(dt_s > 0, dx_km * 1000.0 / dt_safe, 0.0)
    v = np.where(dt_s > 0, dy_km * 1000.0 / dt_safe, 0.0)
    return u, v


def holland_profile(V_m, r):
    """Holland (1980) profile = ChavasWindModel.wind_profile of the reference.

    V(r) = V_m*sqrt((r_m/r)^B * exp(1-(r_m/r)^B)) with the exp(-(r/r0)^4)
    envelope; ``r`` in metres, scalar ``V_m`` in m/s.
    """
    r_safe = np.maximum(np.asarray(r, float), 1.0)
    r_ratio = R_M / r_safe
    exponent = np.clip(1.0 - r_ratio ** HOLLAND_B, -50, 50)
    V = V_m * np.sqrt(r_ratio ** HOLLAND_B * np.exp(exponent))
    if K_SHAPE != 1.0:
        outer = r_safe > 3 * R_M
        V = np.where(outer, V ** K_SHAPE, V)
    V = V * np.exp(-((r_safe / R_0) ** 4))
    V = np.where(np.asarray(r, float) < 1.0, 0.0, V)
    return np.clip(V, 0.0, 150.0)


def G_factor(lat_c):
    """Eq. A2, identical to official tc_wind.py."""
    return min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((lat_c - 35.0) / 10.0)))


def wind_increment(V_sym, lat_c, u_t, v_t):
    """Asymmetric increment vector (A1 translation+shear terms) with the
    official 50% cap, applied locally (increment <= 0.5*V_sym).

    ``V_sym`` may be an array (grid of radii); ``u_t``, ``v_t`` scalars (m/s).
    Returns (U_inc, V_inc) with the same shape as ``V_sym``.
    """
    G = G_factor(lat_c)
    U_raw = G * u_t + 0.1 * U_SHEAR * V_sym / 15.0   # v in m/s (paper Eq. A1)
    V_raw = G * v_t + 0.1 * V_SHEAR * V_sym / 15.0
    mag = np.hypot(U_raw, V_raw)
    mag_fac = np.minimum(1.0, (0.5 * V_sym) / np.maximum(mag, 1e-12))
    return U_raw * mag_fac, V_raw * mag_fac


def wind_speed_on_grid(lon_c, lat_c, V_m, u_t, v_t, lon_g, lat_g,
                       max_distance_km=MAX_DISTANCE_KM):
    """Total wind speed (m/s) on the grid for one storm snapshot.

    v_net = v_axi_vector + capped increment, taking the magnitude (A1);
    V_sym(r) replaces the scalar v as in strike_probability.py.
    """
    dist_km = haversine_km(lon_c, lat_c, lon_g[None, :], lat_g[:, None])

    dlon = wrap180(lon_g[None, :] - lon_c)
    dlat = lat_g[:, None] - lat_c
    azimuth = np.arctan2(dlat, dlon)            # math convention, 0 = East

    V_sym = holland_profile(V_m, (dist_km * 1000.0).ravel()).reshape(dist_km.shape)

    U_inc, V_inc = wind_increment(V_sym, lat_c, u_t, v_t)

    u_net = -V_sym * np.sin(azimuth) + U_inc    # tangential (CCW) + increment
    v_net = V_sym * np.cos(azimuth) + V_inc

    ws = np.hypot(u_net, v_net)
    ws[dist_km > max_distance_km] = 0.0
    return ws


# ═════════════════════════════════════════════════════════════════════════════
# Verification against the paper / official repository
# ═════════════════════════════════════════════════════════════════════════════

def reference_point_wind(lon_c, lat_c, V_m, u_t, v_t, lon_t, lat_t):
    """Literal per-point transcription of A1/A2 + official cap + Holland core.

    Used to verify the vectorised grid version is identical.
    """
    dist_km = float(haversine_km(lon_c, lat_c, lon_t, lat_t))
    r = np.array([dist_km * 1000.0])
    V_sym = float(holland_profile(V_m, r)[0])
    az = float(np.arctan2(lat_t - lat_c, wrap180(lon_t - lon_c)))

    G = min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((lat_c - 35.0) / 10.0)))
    U_raw = G * u_t + 0.1 * U_SHEAR * V_sym / 15.0
    V_raw = G * v_t + 0.1 * V_SHEAR * V_sym / 15.0
    mag = np.hypot(U_raw, V_raw)
    fac = min(1.0, 0.5 * V_sym / max(mag, 1e-12))
    U_inc, V_inc = U_raw * fac, V_raw * fac

    u_net = -V_sym * np.sin(az) + U_inc
    v_net = V_sym * np.cos(az) + V_inc
    return float(np.hypot(u_net, v_net))


def official_vmax(V_m, u_t, v_t, lat_c):
    """Official tc_wind.py analytic result (no radial structure):

    v* = v + min(|G*u_t + 0.1*S*v/15|, 0.5*v),  v in m/s.
    """
    G = G_factor(lat_c)
    U_inc = G * u_t + 0.1 * U_SHEAR * V_m / 15.0
    V_inc = G * v_t + 0.1 * V_SHEAR * V_m / 15.0
    mag = float(np.hypot(U_inc, V_inc))
    fac = min(1.0, 0.5 * V_m / max(mag, 1e-12))
    return V_m + mag * fac


def verify_model():
    """Two checks: (1) vectorised == literal transcription; (2) at r = r_m with
    the optimal azimuth the value reproduces the official analytic v* (up to
    the tiny r0-decay envelope of the Holland profile)."""
    rng = np.random.default_rng(7)

    # ── check 1: random points, vectorised vs literal ────────────────────────
    max_err = 0.0
    for _ in range(400):
        lon_c, lat_c = rng.uniform(-140, -90), rng.uniform(5, 40)
        V_m = rng.uniform(15, 75)                       # m/s
        u_t, v_t = rng.uniform(-10, 10, 2)
        lon_t = lon_c + rng.uniform(-4, 4)
        lat_t = lat_c + rng.uniform(-4, 4)
        got = float(wind_speed_on_grid(lon_c, lat_c, V_m, u_t, v_t,
                                       np.array([lon_t]), np.array([lat_t]),
                                       max_distance_km=1e9)[0])
        ref = reference_point_wind(lon_c, lat_c, V_m, u_t, v_t, lon_t, lat_t)
        if ref > 1e-9:
            max_err = max(max_err, abs(got - ref) / ref)
    print(f"[verify 1] vectorised vs literal A1/A2, 400 pts: "
          f"max rel err = {max_err:.3e}")
    assert max_err < 1e-12, "vectorised model != literal transcription"

    # ── check 2: analytic maximum equals official tc_wind.py formula ────────
    max_err2 = 0.0
    for _ in range(300):
        lat_c = rng.uniform(5, 40)
        V_m = rng.uniform(15, 75)
        u_t, v_t = rng.uniform(-12, 12, 2)
        G = G_factor(lat_c)
        # optimal azimuth: tangential unit vector (-sin th, cos th) aligned
        # with the increment -> place the target point along theta_opt
        U_inc = G * u_t + 0.1 * U_SHEAR * V_m / 15.0
        V_inc = G * v_t + 0.1 * V_SHEAR * V_m / 15.0
        theta_opt = float(np.arctan2(-U_inc, V_inc))
        # point at azimuth theta from the centre: dlon ~ cos(th), dlat ~ sin(th)
        r_km = R_M / 1000.0
        lon_c0 = -110.0
        lon_t = lon_c0 + np.cos(theta_opt) * r_km / 111.32 / np.cos(np.deg2rad(lat_c))
        lat_t = lat_c + np.sin(theta_opt) * r_km / 110.57
        got = float(wind_speed_on_grid(lon_c0, lat_c, V_m, u_t, v_t,
                                       np.array([lon_t]), np.array([lat_t]),
                                       max_distance_km=1e9)[0])
        # expected: V_sym(r_m) + min(|inc(r_m)|, 0.5*V_sym(r_m))
        V_rm = float(holland_profile(V_m, np.array([R_M]))[0])
        U_i = G * u_t + 0.1 * U_SHEAR * V_rm / 15.0
        V_i = G * v_t + 0.1 * V_SHEAR * V_rm / 15.0
        exp = V_rm + min(float(np.hypot(U_i, V_i)), 0.5 * V_rm)
        if exp > 1e-9:
            max_err2 = max(max_err2, abs(got - exp) / exp)
    # residual error comes from placing the test point with a planar
    # degrees->km approximation, not from the model formula itself
    print(f"[verify 2] grid value at (r_m, theta_opt) vs analytic "
          f"V_sym(r_m)+capped|inc|: max rel err = {max_err2:.2e} "
          f"(placement-limited)")
    assert max_err2 < 5e-3, "grid maximum != official analytic formula"
    print("[verify] OK - model matches paper Eq. A1/A2 + official tc_wind.py "
          "(incl. 50% cap, v in m/s in the shear term).")


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble probability
# ═════════════════════════════════════════════════════════════════════════════

def compute_ensemble_probability(lon_tr, lat_tr, vmax_kts, time_h,
                                 thresholds_kt=THRESHOLDS_KT,
                                 grid_res=GRID_RES, lon_pad=7.0, lat_pad=7.0,
                                 sampling_hours=TIME_SAMPLING_HOURS):
    """Member-fraction exceedance probability (Lin et al. 2020 Fig. 2 style)."""
    lon_tr = wrap180(lon_tr)
    lon_all, lat_all = lon_tr.ravel(), lat_tr.ravel()
    valid = np.isfinite(lon_all) & np.isfinite(lat_all)
    lon_grid = np.arange(np.floor(lon_all[valid].min() - lon_pad),
                         np.ceil(lon_all[valid].max() + lon_pad) + grid_res / 2,
                         grid_res)
    lat_grid = np.arange(np.floor(lat_all[valid].min() - lat_pad),
                         np.ceil(lat_all[valid].max() + lat_pad) + grid_res / 2,
                         grid_res)

    exceed = {kt: np.zeros((lat_grid.size, lon_grid.size)) for kt in thresholds_kt}
    n_members_used = 0
    t0 = time.time()

    for m in range(lon_tr.shape[0]):
        lon, lat = lon_tr[m], lat_tr[m]
        V = np.asarray(vmax_kts[m], float) * KT_TO_MS
        ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(V) & (V > 1.0)
        if not ok.any():
            continue
        u_t, v_t = translation_velocity(lon, lat, time_h)

        idx = np.where(ok)[0]
        if sampling_hours > 0:                       # 6-hourly sampling
            keep, last = [], -999.0
            for i in idx:
                if time_h[i] - last >= sampling_hours - 0.1:
                    keep.append(i)
                    last = time_h[i]
            idx = np.asarray(keep)

        hit = {kt: np.zeros(exceed[kt].shape, dtype=bool) for kt in thresholds_kt}
        for i in idx:
            ws = wind_speed_on_grid(lon[i], lat[i], V[i], u_t[i], v_t[i],
                                    lon_grid, lat_grid)
            for kt in thresholds_kt:
                hit[kt] |= ws >= kt * KT_TO_MS
        for kt in thresholds_kt:
            exceed[kt] += hit[kt]
        n_members_used += 1

    prob = {kt: exceed[kt] / max(n_members_used, 1) * 100.0 for kt in thresholds_kt}
    print(f"    grid {lon_grid.size}x{lat_grid.size} @ {grid_res} deg | "
          f"{n_members_used} members | {time.time() - t0:.1f} s")
    for kt in thresholds_kt:
        print(f"      {kt:2d}-kt  max={np.max(prob[kt]):5.1f}%  "
              f"area(>5%)={np.sum(prob[kt] > 5) * grid_res ** 2:7.1f} deg^2")
    return {"lon": lon_grid, "lat": lat_grid, "prob": prob,
            "n_members": n_members_used, "thresholds": list(thresholds_kt)}


def compute_track_strike_probability(lon_tr, lat_tr, time_h,
                                     radius_km=STRIKE_RADIUS_KM,
                                     grid_res=STRIKE_GRID_RES,
                                     pad_deg=STRIKE_PAD_DEG,
                                     max_days=STRIKE_MAX_DAYS):
    """75-km track strike probability, following exactly
    Reproduce/track_model/visualize_strike_prob_75km.py:

    * radius 75 km, 0.2-degree grid, bounds = track extent +/- 1.0 degree;
    * every hourly time step is used (no 6-hourly sampling);
    * 5-day window from initialisation;
    * P = (# tracks whose centre passes within radius) / n_tracks.
    """
    lon_tr = wrap180(lon_tr)
    if max_days is not None:
        tmask = time_h <= max_days * 24.0 + 1e-6
        lon_tr, lat_tr = lon_tr[:, tmask], lat_tr[:, tmask]

    lon_all, lat_all = lon_tr.ravel(), lat_tr.ravel()
    valid = np.isfinite(lon_all) & np.isfinite(lat_all)
    lon_grid = np.arange(np.nanmin(lon_all[valid]) - pad_deg,
                         np.nanmax(lon_all[valid]) + pad_deg + grid_res, grid_res)
    lat_grid = np.arange(np.nanmin(lat_all[valid]) - pad_deg,
                         np.nanmax(lat_all[valid]) + pad_deg + grid_res, grid_res)

    radius_m = radius_km * 1000.0
    strikes = np.zeros((lat_grid.size, lon_grid.size))
    n_tracks = lon_tr.shape[0]
    t0 = time.time()

    for t in range(n_tracks):
        lon_t, lat_t = lon_tr[t], lat_tr[t]
        mask = np.isfinite(lon_t) & np.isfinite(lat_t)
        if not mask.any():
            continue
        lon_t, lat_t = lon_t[mask], lat_t[mask]
        for i, la in enumerate(lat_grid):
            # rows outside this track's reach cannot be struck (fast reject)
            if la < lat_t.min() - 1.0 or la > lat_t.max() + 1.0:
                continue
            d = haversine_km(lon_grid[:, None], la,
                             lon_t[None, :], lat_t[None, :])
            strikes[i] += np.any(d <= radius_km, axis=1)
        if (t + 1) % 100 == 0 or t == n_tracks - 1:
            print(f"      [{t + 1}/{n_tracks}] strikes max={strikes.max()}")

    prob = strikes / n_tracks * 100.0
    window_s = f"{max_days:.0f}-day window" if max_days is not None else "full window"
    print(f"    grid {lon_grid.size}x{lat_grid.size} @ {grid_res} deg | "
          f"{n_tracks} tracks | r={radius_km:.0f} km | "
          f"{window_s} | {time.time() - t0:.1f} s")
    print(f"      strike  max={np.max(prob):5.1f}%  "
          f"area(>5%)={np.sum(prob > 5) * grid_res ** 2:7.1f} deg^2  "
          f"area(>50%)={np.sum(prob > 50) * grid_res ** 2:7.1f} deg^2")
    return {"lon": lon_grid, "lat": lat_grid, "prob": prob,
            "n_members": n_tracks}


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

PROB_COLORS = ['#F0F8FF', '#E0F0FF', '#B0D8FF', '#80C0FF', '#50A8FF',
               '#40C0B0', '#60D880', '#80F060', '#A0FF40',
               '#D0FF20', '#FFE800', '#FFD000', '#FFB000',
               '#FF8000', '#FF4000', '#E00000', '#C00000']
PROB_LEVELS = np.linspace(0, 100, 51)


def _base_map(ax, lon_grid, lat_grid):
    """Geographic base exactly as Reproduce/strike_probability.py:
    coastlines lw=0.8 zorder=10, borders lw=0.4 gray, LAND '#F5F5DC' alpha=0.4
    zorder=5, OCEAN '#E8F4F8' alpha=0.3 zorder=4, 10% extent buffer."""
    lon_buffer = (lon_grid[-1] - lon_grid[0]) * 0.1
    lat_buffer = (lat_grid[-1] - lat_grid[0]) * 0.1
    ax.set_extent([lon_grid[0] - lon_buffer, lon_grid[-1] + lon_buffer,
                   lat_grid[0] - lat_buffer, lat_grid[-1] + lat_buffer],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#E8F4F8",
                   alpha=0.3, zorder=4)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F5F5DC",
                   alpha=0.4, zorder=5)
    ax.coastlines("110m", linewidth=0.8, color="black", zorder=10)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.4,
                   edgecolor="gray", zorder=10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False


def _mean_track(tracks):
    lon180 = wrap180(tracks[0])
    mean_lon = np.nanmean(lon180, axis=0)
    mean_lat = np.nanmean(tracks[1], axis=0)
    return mean_lon, mean_lat


def plot_probability_maps(prob, storm, init_time, model_label, out_path,
                          tracks=None, dpi=170):
    """Wind-speed exceedance probability panels (strike_probability.py style).

    If ``tracks`` is given (raw 31-member GEFS), they are drawn as thin black
    spaghetti (no ensemble mean), and every panel is zoomed to the track
    strike map domain (track extent +/- 1 degree) instead of the full
    wind-probability grid.
    """
    thresholds = prob["thresholds"]
    ncols, nrows = 3, int(np.ceil(len(thresholds) / 3))
    # figure geometry identical to Reproduce/strike_probability.py
    fig = plt.figure(figsize=(22, 7 * nrows), dpi=dpi)
    gs = GridSpec(nrows, ncols + 1, figure=fig,
                  width_ratios=[1] * ncols + [0.08], hspace=0.3, wspace=0.2)
    cmap = LinearSegmentedColormap.from_list("prob_cmap", PROB_COLORS, N=100)
    lon_grid, lat_grid = prob["lon"], prob["lat"]

    # zoom domain = track-strike domain (track extent +/- 1 degree)
    lon180 = wrap180(tracks[0]) if tracks is not None else lon_grid
    lat_tr = tracks[1] if tracks is not None else lat_grid
    zoom = [np.nanmin(lon180) - 1.0, np.nanmax(lon180) + 1.0,
            np.nanmin(lat_tr) - 1.0, np.nanmax(lat_tr) + 1.0]

    cf = None
    for idx, kt in enumerate(thresholds):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols],
                             projection=ccrs.PlateCarree())
        p = prob["prob"][kt]
        cf = ax.contourf(lon_grid, lat_grid, p, levels=PROB_LEVELS, cmap=cmap,
                         transform=ccrs.PlateCarree(), extend="neither",
                         antialiased=True)
        ax.contour(lon_grid, lat_grid, p,
                   levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
                   colors="black", linewidths=0.3, alpha=0.3,
                   transform=ccrs.PlateCarree())
        _base_map(ax, lon_grid, lat_grid)

        if tracks is not None:                     # raw GEFS spaghetti only
            for m in range(lon180.shape[0]):
                mask = np.isfinite(lon180[m]) & np.isfinite(lat_tr[m])
                ax.plot(lon180[m, mask], lat_tr[m, mask], color="black",
                        lw=0.5, alpha=0.35, transform=ccrs.PlateCarree(),
                        zorder=6)
            ax.set_extent(zoom, crs=ccrs.PlateCarree())
        ax.set_title(f"{kt}-kt Wind Speed Probability", fontsize=14,
                     fontweight="bold", pad=10)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(cf, cax=cax, orientation="vertical",
                        ticks=[0, 15, 30, 45, 60, 75, 90, 100])
    cbar.set_label("Probability (%)", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(f"{storm} ({model_label}) Wind Speed Exceedance Probability\n"
                 f"Initialized: {init_time}",
                 fontsize=18, fontweight="bold", y=0.98)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} (+.svg)")


STRIKE_COLORS = ["#ffffff", "#cfe6f5", "#8dc1e0", "#5aa2c9", "#2f7f9f", "#c59d55", "#7f9e4c", "#549c68", "#2f834a", "#5b8150",
                "#b06f2c", "#d06b3a", "#c05f77", "#d58ea7", "#d6967c", "#daa7a3", "#e1b0b7", "#e7c1d0", "#c8acbd", "#8f8c8c", "#666666"]


def plot_track_strike_map(strike, tracks, storm, init_time, out_path,
                          n_spaghetti=200, dpi=170):
    """75-km track strike-probability map, copied verbatim from
    Reproduce/track_model/visualize_strike_prob_75km.py::plot_strike_probability:
    white land/ocean, 21-colour ListedColormap, levels 0-100 step 5 with
    extend='max', alpha=0.9 zorder=2, black spaghetti lw=0.5 alpha=0.35
    zorder=3, coastline/borders zorder=4, colorbar fraction=0.046 pad=0.04
    with ticks every 10%, title fontsize 13 bold.

    ``tracks`` = (lon, lat) of the raw 31-member GEFS ensemble, drawn as the
    thin black spaghetti (the reference plots lon_orig/lat_orig here); the
    probability field itself is computed from the full sampled ensemble.
    """
    from matplotlib.colors import ListedColormap

    lon_grid, lat_grid = strike["lon"], strike["lat"]
    prob = strike["prob"]
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": proj},
                           dpi=dpi)
    ax.add_feature(cfeature.LAND, facecolor="white", alpha=1.0, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=1.0, zorder=0)

    cf = ax.contourf(lon_grid, lat_grid, prob, levels=np.arange(0, 105, 5),
                     cmap=ListedColormap(STRIKE_COLORS), extend="max",
                     transform=proj, alpha=0.9, zorder=2, antialiased=True)

    # raw GEFS members as thin black spaghetti (reference lon_orig/lat_orig)
    lon_o, lat_o = tracks
    lon_o = wrap180(lon_o)
    for i in range(min(n_spaghetti, lon_o.shape[0])):
        mask = np.isfinite(lon_o[i]) & np.isfinite(lat_o[i])
        ax.plot(lon_o[i, mask], lat_o[i, mask], color="black",
                linewidth=0.5, alpha=0.35, transform=proj, zorder=3)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=4)
    ax.set_facecolor("white")
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5,
                 linestyle="--")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04,
                        ticks=np.arange(0, 105, 10))
    cbar.set_label("Strike Probability (%)")
    ax.set_title(f"{storm} 75-km Strike Probability "
                 f"(FAST-ML ensemble {strike['n_members']} members)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_svg = out_path.with_suffix(".svg")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} (+.svg)")


def plot_tracks(lon_tr, lat_tr, storm, model_label, init_time, out_path,
                n_spaghetti=300, dpi=170):
    """Spaghetti track plot, styling copied from
    Reproduce/plot_fast_results_irma.py (gray '#B0B0B0' members lw=0.8
    alpha=0.6, LAND '0.95'/OCEAN '0.98' alpha=0.8, title
    fontsize 18). Only raw GEFS members - no ensemble mean, per user request."""
    lon180 = wrap180(lon_tr)

    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    pad = 3.0
    ax.set_extent([np.nanmin(lon180) - pad, np.nanmax(lon180) + pad,
                   np.nanmin(lat_tr) - pad, np.nanmax(lat_tr) + pad],
                  crs=ccrs.PlateCarree())
    # base map exactly as plot_fast_results_irma.py
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6,
                   zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.4, zorder=4)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="0.95",
                   alpha=0.8, zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="0.98",
                   alpha=0.8, zorder=1)

    for m in range(min(n_spaghetti, lon180.shape[0])):
        mask = np.isfinite(lon180[m]) & np.isfinite(lat_tr[m])
        ax.plot(lon180[m, mask], lat_tr[m, mask], color="#B0B0B0",
                linewidth=0.8, alpha=0.6, transform=ccrs.PlateCarree(), zorder=3)
    # no ensemble-mean track: draw only the raw GEFS members
    ax.plot(lon180[0, 0], lat_tr[0, 0], "k*", ms=15,
            transform=ccrs.PlateCarree(), zorder=9, label="Init")

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray",
                      alpha=0.5, linestyle="--", zorder=5)
    gl.top_labels = False
    gl.right_labels = False
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)
    ax.set_title(f"Track Trajectories ({storm}, raw GEFS {lon_tr.shape[0]} members)",
                 fontsize=18, fontweight="bold", pad=20)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name}")


def save_prob_netcdf(prob_by_model, storm, out_path):
    """Save the wind-speed probability fields (0.25-degree grid)."""
    import xarray as xr
    data_vars = {}
    for model, prob in prob_by_model.items():
        for kt in prob["thresholds"]:
            data_vars[f"prob_{kt}kt_{model}"] = (
                ("lat", "lon"), prob["prob"][kt],
                {"long_name": f"{kt}-kt exceedance probability ({model})",
                 "units": "percent"})
    ref = next(iter(prob_by_model.values()))
    ds = xr.Dataset(
        data_vars=data_vars, coords={"lon": ref["lon"], "lat": ref["lat"]},
        attrs={"description": f"{storm} wind speed exceedance probability",
               "definition": "fraction of members whose modelled wind exceeds "
                             "the threshold at least once (6-hourly samples)",
               "wind_model": "Eq. A1/A2 of Lin et al. (2023) / official "
                             "tc_wind.py (v in m/s, 50% increment cap) on a "
                             "Holland B=1.5 radial profile",
               "r_m_m": R_M, "r_0_m": R_0, "k": K_SHAPE,
               "default_shear_ms": f"({U_SHEAR}, {V_SHEAR})",
               "grid_resolution_degrees": GRID_RES,
               "time_sampling_hours": TIME_SAMPLING_HOURS})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    ds.close()
    print(f"  -> {out_path.name}")


def save_strike_netcdf(strike, storm, out_path):
    """Save the 75-km strike probability on its own 0.2-degree grid."""
    import xarray as xr
    ds = xr.Dataset(
        data_vars={
            "prob_strike": (("lat", "lon"), strike["prob"],
                            {"long_name": f"{STRIKE_RADIUS_KM:.0f}-km track "
                             "strike probability",
                             "units": "percent"}),
        },
        coords={"lon": strike["lon"], "lat": strike["lat"]},
        attrs={"description": f"{storm} track strike probability",
               "definition": "fraction of tracks whose centre passes within "
                             f"{STRIKE_RADIUS_KM:.0f} km, following "
                             "Reproduce/track_model/visualize_strike_prob_75km.py",
               "radius_km": STRIKE_RADIUS_KM,
               "grid_resolution_degrees": STRIKE_GRID_RES,
               "max_days": STRIKE_MAX_DAYS})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    ds.close()
    print(f"  -> {out_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════════

def run_case(name, cfg, model_sel, data_dir, out_dir):
    import xarray as xr

    print(f"\n=== {cfg['storm']}  (init {cfg['init_time']}) ===")
    nc = data_dir / cfg["ode_nc"]
    with xr.open_dataset(nc) as ds:
        lon = ds["lon"].values
        lat = ds["lat"].values
        time_h = ds["time_hours"].values
        vmax = {key: ds[var].values for key, var in MODEL_KEYS.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw 31-member GEFS tracks (pre-sampling) for drawing, as the reference
    # script does via lon_orig/lat_orig
    raw_pkl = RAW_TRACKS.get(name)
    lon_raw, lat_raw = (None, None)
    if raw_pkl and raw_pkl.exists():
        lon_raw, lat_raw = load_raw_gefs_tracks(raw_pkl, max_days=STRIKE_MAX_DAYS)
        if lon_raw is not None:
            print(f"  loaded raw GEFS tracks: {lon_raw.shape[0]} members "
                  f"from {raw_pkl.name}")
    draw_tracks = (lon_raw, lat_raw) if lon_raw is not None else (lon, lat)

    # ── Figure 1: 75-km track strike probability (shared GEFS tracks) ───────
    print("  [Track strike probability]")
    strike = compute_track_strike_probability(lon, lat, time_h)
    plot_track_strike_map(strike, draw_tracks, cfg["storm"], cfg["init_time"],
                          out_dir / f"track_strike_{name}")
    save_strike_netcdf(strike, cfg["storm"],
                       out_dir / f"strike_prob_75km_{name}.nc")

    # ── Figure 2: wind-speed exceedance probability per model ──────────────
    prob_by_model = {}
    for key in model_sel:
        label = MODEL_LABELS[key]
        print(f"  [{label}]")
        prob = compute_ensemble_probability(lon, lat, vmax[key], time_h,
                                            lon_pad=cfg["lon_pad"],
                                            lat_pad=cfg["lat_pad"])
        prob_by_model[key] = prob
        plot_probability_maps(prob, cfg["storm"], cfg["init_time"], label,
                              out_dir / f"strike_probability_{name}_{key}",
                              tracks=draw_tracks)
        plot_tracks(*draw_tracks, cfg["storm"], label, cfg["init_time"],
                    out_dir / f"tracks_{name}_{key}.png", n_spaghetti=31)

    save_prob_netcdf(prob_by_model, cfg["storm"],
                     out_dir / f"strike_probability_{name}.nc")


def main():
    parser = argparse.ArgumentParser(
        description="FAST / FAST-ML wind-speed exceedance probability maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--case", nargs="+", choices=sorted(CASES) + ["all"],
                        default=["all"])
    parser.add_argument("--model", choices=["ml", "fast", "both"], default="ml",
                        help="ml = FAST-ML predictions (default), fast = FAST")
    parser.add_argument("--data-dir", type=Path, default=ENSEMBLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "ensemble")
    parser.add_argument("--verify", action="store_true",
                        help="check the wind model against paper Eq. A1/A2 "
                             "and the official tc_wind.py, then exit")
    args = parser.parse_args()

    if args.verify:
        verify_model()
        return 0

    model_sel = ["fast", "ml"] if args.model == "both" else [args.model]
    names = sorted(CASES) if "all" in args.case else args.case
    for name in names:
        run_case(name, CASES[name], model_sel, args.data_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
