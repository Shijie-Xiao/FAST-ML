#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAST intensity ODE, integrated in NumPy.

This is the deterministic physics core shared by both experiments in this
repository. FAST and FAST_ML differ *only* in where the ventilation index
``chi * S`` comes from:

* FAST    -- ERA5 reanalysis diagnostics (``chi_ref``, ``s_ref``)
* FAST_ML -- the two-stream CNN in :mod:`fastml.model`

Everything downstream of ``chi * S`` (the coupled (V, m) ODE, the 48 h
initialisation, the forcing decay, and the axisymmetric-to-surface wind
conversion) is byte-for-byte identical between the two, so any difference in
the predicted intensity is attributable to the ventilation term alone.

Governing equations (Emanuel 2012; ventilation after Tang & Emanuel 2012,
Lin et al. 2023), with ``coeff = 0.5 * Cd / h_bl * 3600``:

    dV/dt = coeff * (alpha * beta * vp^2 * m^3 - (1 - gamma * m^3) * V^2)
    dm/dt = coeff * ((1 - m) * V - chi * S * m)
"""

from __future__ import annotations

import numpy as np

from .config import (
    BASIN_BOUNDS,
    CD_CONST,
    CHI_D_ATLANTIC,
    CHI_MULTIPLIER,
    H_BL,
    INIT_HOURS,
    STEP_SIZE,
    SUB_STEPS,
    T0_DECAY_HOURS,
    VMAX_START_MS,
    XS_NAN_FALLBACK,
    cd_nc_path,
)

# Cached Cd interpolator so repeated storm runs do not re-read Cd.nc.
_F_CD = None
_F_CD_FAILED = False


def _adj_bound(bound: str) -> float:
    """``'260E' -> 260.0``, ``'45S' -> -45.0``. Mirrors ``util.basins``."""
    value = float(bound[:-1])
    return -value if bound[-1] in ("W", "S") else value


def load_cd_interpolator(basin: str = "NA"):
    """Build the spatially varying drag-coefficient interpolator for a basin.

    Reproduces ``tropical_cyclone_risk.intensity.geo.read_drag`` without
    requiring that package: the 10 m drag coefficient is converted to a
    gradient-wind drag coefficient, normalised by its over-ocean minimum, and
    rescaled by the neutral value ``Cd``. Land therefore carries a much larger
    Cd, which is what makes storms spin down after landfall.

    Returns ``None`` if ``Cd.nc`` is unavailable, in which case callers fall
    back to the constant ``CD_CONST``.
    """
    global _F_CD, _F_CD_FAILED
    if _F_CD is not None or _F_CD_FAILED:
        return _F_CD

    try:
        import xarray as xr
        from scipy.interpolate import RectBivariateSpline

        path = cd_nc_path()
        with xr.open_dataset(path) as ds:
            lon = ds["longitude"].data
            lat = ds["latitude"].data
            cd = ds["Cd"].data

        cd_gradient = cd / (1.0 + 250.0 * cd)
        cd_norm = cd_gradient / np.min(cd_gradient)
        cd = CD_CONST * cd_norm

        lon_min, lat_min, lon_max, lat_max = (
            _adj_bound(b) for b in BASIN_BOUNDS[basin]
        )
        lon_mask = (lon <= lon_max + 1e-5) & (lon >= lon_min - 1e-5)
        lat_mask = (lat >= lat_min - 1e-5) & (lat <= lat_max + 1e-5)
        cd_b = cd[lat_mask, :][:, lon_mask]

        _F_CD = RectBivariateSpline(lon[lon_mask], lat[lat_mask], cd_b.T, kx=1, ky=1)
    except Exception as exc:  # pragma: no cover - degraded but still runnable
        print(f"  [fast_physics] Cd.nc unavailable ({exc}); using constant Cd={CD_CONST}")
        _F_CD_FAILED = True
        _F_CD = None
    return _F_CD


def get_cd_at(lon, lat, f_cd) -> float:
    """Interpolate Cd at one point, with longitude wrapped to 0-360."""
    if f_cd is None:
        return CD_CONST
    lon = float(lon) if np.isfinite(lon) else 0.0
    lat = float(lat) if np.isfinite(lat) else 0.0
    if lon < 0:
        lon += 360
    try:
        return float(f_cd.ev(lon, lat).flatten()[0])
    except Exception:
        return CD_CONST


def coeff_from_cd(cd, h_bl=H_BL) -> float:
    """``0.5 * Cd / h_bl * 3600`` -- units of (m/s) per hour."""
    return 0.5 * float(cd) / float(h_bl) * 3600.0


def coeff_series(lons, lats, T, basin="NA") -> np.ndarray:
    """Per-timestep ODE coefficient along a storm track."""
    f_cd = load_cd_interpolator(basin)
    lo = np.zeros(T)
    la = np.zeros(T)
    if lons is not None:
        a = np.asarray(lons).reshape(-1)
        lo[: min(len(a), T)] = a[:T]
    if lats is not None:
        a = np.asarray(lats).reshape(-1)
        la[: min(len(a), T)] = a[:T]
    return np.array(
        [coeff_from_cd(get_cd_at(lo[t], la[t], f_cd)) for t in range(T)],
        dtype=np.float64,
    )


def median_filter_1d(arr, size=3) -> np.ndarray:
    """3 h running median, used to damp hour-to-hour jitter in ``vp``."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr
    padded = np.pad(arr, 1, mode="edge")
    return np.array([np.median(padded[i : i + size]) for i in range(n)], dtype=np.float64)


def chi_calibrated_multiply(chi_val, chi_multiplier=CHI_MULTIPLIER, chi_max=CHI_D_ATLANTIC):
    """Scale the mean ventilation index up to a ~90th-percentile 'driest air' value.

    The storm-relative dry-air distribution is roughly log-normal, so its 90th
    percentile sits a few times above the annulus mean; a constant multiplier
    clipped at the Atlantic physical ceiling reproduces that without the
    per-storm fitting used in the original formulation.
    """
    chi_val = np.asarray(chi_val, dtype=np.float64)
    chi_val = np.nan_to_num(chi_val, nan=1e-10)
    chi_val = np.maximum(chi_val, 1e-10)
    return np.clip(chi_val * chi_multiplier, 0.0, chi_max)


def _safe_scalar(x, default) -> float:
    """Replace NaN by a default, without clipping to an assumed range."""
    return float(default) if np.isnan(x) else float(x)


def _physics_rhs_v(V, m, alpha, beta, gamma, vp, coeff) -> float:
    """dV/dt of the FAST system."""
    vp = _safe_scalar(vp, 0.0)
    alpha = _safe_scalar(alpha, 1.0)
    beta = _safe_scalar(beta, 0.57)
    gamma = _safe_scalar(gamma, 0.43)
    m3 = m ** 3
    with np.errstate(invalid="ignore", divide="ignore"):
        rhs = coeff * (alpha * beta * vp ** 2 * m3 - (1.0 - gamma * m3) * V ** 2)
    return rhs if np.isfinite(rhs) else 0.0


def _physics_rhs_m(V, m, xs, coeff) -> float:
    """dm/dt of the FAST system."""
    xs = _safe_scalar(xs, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rhs = coeff * ((1.0 - m) * V - xs * m)
    return rhs if np.isfinite(rhs) else 0.0


def fast_step(xs, V, m, alpha, beta, gamma, vp, coeff, dV_extra=0.0):
    """One Heun (improved Euler) sub-step of the coupled (V, m) system."""
    vp = _safe_scalar(vp, 0.0)
    alpha = _safe_scalar(alpha, 1.0)
    beta = _safe_scalar(beta, 0.57)
    gamma = _safe_scalar(gamma, 0.43)
    xs = _safe_scalar(xs, 0.0)

    m3 = m ** 3
    dV = coeff * (alpha * beta * vp ** 2 * m3 - (1.0 - gamma * m3) * V ** 2) + dV_extra
    dm = coeff * ((1.0 - m) * V - xs * m)

    V_mid = max(0.0, min(200.0, V + dV * STEP_SIZE))
    m_mid = max(0.0, min(1.0, m + dm * STEP_SIZE))

    m3_mid = m_mid ** 3
    dV2 = coeff * (alpha * beta * vp ** 2 * m3_mid - (1.0 - gamma * m3_mid) * V_mid ** 2) + dV_extra
    dm2 = coeff * ((1.0 - m_mid) * V_mid - xs * m_mid)

    V_next = max(0.0, min(200.0, V + 0.5 * (dV + dV2) * STEP_SIZE))
    m_next = max(0.0, min(1.0, m + 0.5 * (dm + dm2) * STEP_SIZE))
    return V_next, m_next


def calculate_m0_from_fast(v, dv_dt, alpha, beta, gamma, vp, coeff):
    """Invert dV/dt for the initial moisture variable m.

    ``m^3 = (dV/dt / coeff + V^2) / (alpha * beta * vp^2 + gamma * V^2)``
    """
    vp = _safe_scalar(vp, 0.0)
    alpha = _safe_scalar(alpha, 1.0)
    beta = _safe_scalar(beta, 0.57)
    gamma = _safe_scalar(gamma, 0.43)
    numerator = dv_dt / (coeff + 1e-12) + v ** 2
    denominator = alpha * beta * vp ** 2 + gamma * v ** 2
    m3 = np.clip(numerator / (denominator + 1e-8), 0.0, None)
    return np.clip(np.power(m3, 1.0 / 3.0), 0.01, 1.0)


def _axi_to_max_wind_single(v_axisym, s, env, ut, vt, lat) -> float:
    """Scalar version of :func:`axi_to_max_wind`, used by the inversion."""
    v_axisym = float(np.asarray(v_axisym).flat[0])
    s = float(np.nan_to_num(np.asarray(s).flat[0], nan=0.0))
    env = np.atleast_2d(env)
    if env.shape[0] == 1:
        env = env.reshape(4)
    ut = float(np.asarray(ut).flat[0])
    vt = float(np.asarray(vt).flat[0])
    lat = float(np.asarray(lat).flat[0])

    G = min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((abs(lat) - 35.0) / 10.0)))
    u_shr = env[0] - env[2]
    v_shr = env[1] - env[3]
    shear_mag = np.sqrt(u_shr ** 2 + v_shr ** 2 + 1e-12)
    has_env = not (np.isnan(u_shr) or np.isnan(v_shr) or shear_mag < 1e-6)
    u_dir = (u_shr / shear_mag) if has_env else 0.0
    v_dir = (v_shr / shear_mag) if has_env else 0.0

    shear_coeff = 0.1 * s * v_axisym / 15.0
    U_inc = G * ut + shear_coeff * u_dir
    V_inc = G * vt + shear_coeff * v_dir
    mag_inc = np.sqrt(U_inc ** 2 + V_inc ** 2 + 1e-12)
    mag_fac = min(1.0, (v_axisym * 0.5) / mag_inc) if mag_inc > 1e-12 else 0.0

    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = v_axisym * (-np.sin(theta_opt)) + U_inc * mag_fac
    vg = v_axisym * np.cos(theta_opt) + V_inc * mag_fac
    return float(np.sqrt(ug ** 2 + vg ** 2 + 1e-12))


def invert_vmax_to_v_axisym(v_max_obs, s, env, ut, vt, lat,
                            v_lo=0.0, v_hi=200.0, tol=0.01, max_iter=50):
    """Bisect :func:`_axi_to_max_wind_single` to recover axisymmetric V from vmax.

    The observed maximum sustained wind contains the storm translation and
    shear-induced asymmetry; the ODE state variable is the axisymmetric mean,
    so the observation has to be inverted before it can be used as a target.
    """
    if np.isnan(v_max_obs) or v_max_obs <= 0:
        return np.nan
    v_max_obs = float(v_max_obs)
    for _ in range(max_iter):
        v_mid = (v_lo + v_hi) * 0.5
        v_max_mid = _axi_to_max_wind_single(v_mid, s, env, ut, vt, lat)
        if abs(v_max_mid - v_max_obs) < tol:
            return v_mid
        if v_max_mid < v_max_obs:
            v_lo = v_mid
        else:
            v_hi = v_mid
    return (v_lo + v_hi) * 0.5


def axi_to_max_wind(tc_v, s_ref, env_wnds, utran, vtran, lats) -> np.ndarray:
    """Convert axisymmetric wind to maximum surface wind, vectorised over time.

    Adds the translation speed (latitude-dependent gust factor ``G``) and a
    shear-aligned asymmetry whose magnitude scales with ``S``, then takes the
    wind speed at the optimal azimuth.
    """
    G = np.minimum(1.0, 0.8 + 0.35 * (1.0 + np.tanh((np.abs(lats) - 35.0) / 10.0)))
    u_shr = env_wnds[:, 0] - env_wnds[:, 2]
    v_shr = env_wnds[:, 1] - env_wnds[:, 3]
    shear_mag = np.sqrt(u_shr ** 2 + v_shr ** 2 + 1e-12)
    has_env = ~(np.isnan(u_shr) | np.isnan(v_shr) | (shear_mag < 1e-6))
    u_dir = np.where(has_env, u_shr / shear_mag, 0.0)
    v_dir = np.where(has_env, v_shr / shear_mag, 0.0)

    s_safe = np.nan_to_num(s_ref, nan=0.0)
    shear_coeff = 0.1 * s_safe * tc_v / 15.0
    U_inc = G * utran + shear_coeff * u_dir
    V_inc = G * vtran + shear_coeff * v_dir
    mag_inc = np.sqrt(U_inc ** 2 + V_inc ** 2 + 1e-12)
    mag_fac = np.minimum(1.0, (tc_v * 0.5) / mag_inc)

    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = tc_v * (-np.sin(theta_opt)) + U_inc * mag_fac
    vg = tc_v * np.cos(theta_opt) + V_inc * mag_fac
    return np.sqrt(ug ** 2 + vg ** 2 + 1e-12)


def _as_time_series(arr, T, width=None, fill=0.0):
    """Coerce ``[T]``/``[1,T]``/``[1,T,1]``/``[1,T,W]`` inputs to ``[T]`` or ``[T,W]``."""
    if width is None:
        out = np.full(T, fill, dtype=np.float64)
        if arr is None:
            return out
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        out[: min(len(a), T)] = a[:T]
        return out

    out = np.full((T, width), np.nan)
    if arr is None:
        return out
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        return np.array(a[0, :T, :], dtype=np.float64)
    out[: min(a.shape[0], T), :] = a[:T, :]
    return out


def find_t_start(v_obz, T):
    """Forecast start index and initialisation window, from the 45 kt threshold.

    The forecast begins when the observed intensity first reaches 45 kt, and the
    preceding 48 h are used to initialise the ODE. If fewer than 48 h of history
    exist, the start is pushed back to t = 48 instead.
    """
    t_45 = None
    for i in range(T):
        if not np.isnan(v_obz[i]) and v_obz[i] >= VMAX_START_MS:
            t_45 = i
            break
    if t_45 is None:
        t_45 = 0

    t_start = t_45
    t_init_start = t_start - INIT_HOURS
    if t_init_start < 0:
        t_start = INIT_HOURS
        t_init_start = 0
    if t_start > T:
        t_start = T
        t_init_start = max(0, T - INIT_HOURS)
    return t_start, t_init_start


def run_fast_with_init(scalars, xs, v_gt, env_wnds, utran, vtran, lats, s_ref, lons=None):
    """Integrate FAST with a 48 h observation-nudged initialisation.

    Phases:

    1. Locate the forecast start ``t_start`` (first time the observation
       reaches 45 kt) and the initialisation window ``[t_init_start, t_start)``.
    2. Over the initialisation window, nudge V onto the inverted observation
       ``Vtarget`` and diagnose the residual forcing
       ``F(t) = observed acceleration - physics RHS``, letting m spin up
       naturally. ``F_init_end`` is the mean F over the final 12 h.
    3. From ``t_start`` onward, integrate freely with the stored forcing decaying
       as ``F_init_end * exp(-2 * (lead / 24 h)^2)``, so the storm's inherited
       momentum fades rather than vanishing at the forecast start.

    Args:
        scalars: ``[1, T, 4]`` of (alpha, beta, gamma, vp), vp in m/s.
        xs: ``[1, T, 1]`` ventilation index ``chi * S``. This is the *only*
            input that differs between FAST and FAST_ML.
        v_gt: ``[1, T, 1]`` observed maximum sustained wind, m/s.
        env_wnds: ``[1, T, 4]`` of (u250, v250, u850, v850), m/s.
        utran, vtran: ``[1, T, 1]`` storm translation velocity, m/s.
        lats, lons: ``[T]`` or ``[1, T]`` track coordinates, degrees.
        s_ref: ``[T]``-like ventilation shear term, used for the wind conversion.

    Returns:
        ``(v_axisym, v_max, m)`` each ``[T]`` in m/s (m is dimensionless), with
        NaN before ``t_init_start``.
    """
    T = scalars.shape[1]
    scalars_np = np.array(scalars[0, :, :], dtype=np.float64)
    vp_used = scalars_np[:, 3]
    alpha, beta, gamma = scalars_np[:, 0], scalars_np[:, 1], scalars_np[:, 2]
    xs_np = np.maximum(
        np.nan_to_num(np.array(xs[0, :, 0], dtype=np.float64), nan=XS_NAN_FALLBACK),
        XS_NAN_FALLBACK,
    )
    v_obz = np.array(v_gt[0, :, 0], dtype=np.float64)

    la = _as_time_series(lats, T)
    lo = _as_time_series(lons, T)
    coeff_arr = coeff_series(lo, la, T)

    ew = _as_time_series(env_wnds, T, width=4)
    ut = _as_time_series(utran, T)
    vt = _as_time_series(vtran, T)
    s_r = (
        np.nan_to_num(np.asarray(s_ref).reshape(T, -1)[:, 0], nan=0.0)
        if s_ref is not None
        else np.zeros(T)
    )

    t_start, t_init_start = find_t_start(v_obz, T)

    # Invert the observation to the axisymmetric target the ODE actually evolves.
    Vtarget = np.full(T, np.nan)
    for i in range(T):
        if np.isnan(v_obz[i]) or v_obz[i] <= 0:
            continue
        Vtarget[i] = invert_vmax_to_v_axisym(v_obz[i], s_r[i], ew[i], ut[i], vt[i], la[i])

    v_fast = np.full(T, np.nan)
    m_series = np.full(T, np.nan)
    if t_init_start >= T:
        return v_fast, np.full(T, np.nan), m_series

    v0 = float(Vtarget[t_init_start]) if not np.isnan(Vtarget[t_init_start]) else 5.0
    if v0 <= 0:
        v0 = 5.0
    V = np.float64(v0)

    dv_dt = 0.0
    if (
        t_init_start + 1 < T
        and not np.isnan(Vtarget[t_init_start])
        and not np.isnan(Vtarget[t_init_start + 1])
    ):
        dv_dt = Vtarget[t_init_start + 1] - Vtarget[t_init_start]
    m0 = calculate_m0_from_fast(
        v0, dv_dt,
        alpha[t_init_start], beta[t_init_start], gamma[t_init_start],
        vp_used[t_init_start], coeff_arr[t_init_start],
    )
    m = np.float64(np.clip(m0, 0.01, 1.0))

    F_init_end = 0.0
    F_history = []

    for t in range(t_init_start, T):
        coeff_t = coeff_arr[t]
        if t < t_start:
            # Initialisation: track the observation, diagnose the residual forcing.
            Vtar_t = float(Vtarget[t]) if not np.isnan(Vtarget[t]) else V
            Vtar_next = (
                float(Vtarget[t + 1])
                if t + 1 < T and not np.isnan(Vtarget[t + 1])
                else Vtar_t
            )
            observed_accel = Vtar_next - Vtar_t
            physics_rhs = _physics_rhs_v(
                Vtar_t, m, alpha[t], beta[t], gamma[t], vp_used[t], coeff_t
            )
            F_history.append(observed_accel - physics_rhs)
            if t == min(t_start, T) - 1:
                window = min(12, len(F_history))
                F_init_end = float(np.mean(F_history[-window:]))
            V = np.float64(Vtar_next)
            for _ in range(SUB_STEPS):
                dm = _physics_rhs_m(Vtar_next, m, xs_np[t], coeff_t)
                m = np.float64(np.clip(m + dm * STEP_SIZE, 0.01, 1.0))
            m_series[t] = float(m)
        else:
            # Forecast: free integration plus the decaying inherited forcing.
            lead_h = t - t_start
            dV_extra = F_init_end * np.exp(-2.0 * (lead_h / T0_DECAY_HOURS) ** 2)
            for _ in range(SUB_STEPS):
                V, m = fast_step(
                    xs_np[t], V, m,
                    alpha[t], beta[t], gamma[t], vp_used[t], coeff_t,
                    dV_extra=dV_extra,
                )
        v_fast[t] = float(V)
        m_series[t] = float(m)

    v_max = axi_to_max_wind(v_fast, s_r, ew, ut, vt, la)
    return v_fast, v_max, m_series
