#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure generation for the single-track and ensemble comparisons.

Both figures share a layout: intensity on top, the ventilation index ``chi * S``
that drives it below, with FAST in green and FAST_ML in blue throughout.
"""

from __future__ import annotations

from pathlib import Path

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    COLOR_FAST,
    COLOR_FAST_ENS,
    COLOR_FASTML,
    COLOR_FASTML_ENS,
    COLOR_GOOGLE,
    COLOR_OBS,
)


# ── Single-track figure ──────────────────────────────────────────────────────

def _last_strong_index(series, i0, min_intensity_kts, peak_drop_frac):
    """Last index at or after ``i0`` exceeding a fraction of the series peak."""
    v = np.asarray(series, dtype=float)
    valid = np.isfinite(v)
    masked = np.where(valid, v, -np.inf)
    peak = float(masked[i0:].max()) if i0 < v.size else -np.inf
    if not np.isfinite(peak):
        return None
    above = valid & (v > max(float(min_intensity_kts), peak * float(peak_drop_frac)))
    return int(np.where(above)[0][-1]) if above.any() else None


def find_plot_window(v_obz_kts, min_intensity_kts=30.0, peak_drop_frac=0.4,
                     tail_buffer_hours=24, model_kts=None):
    """Choose the plotting window ``[i0, i1)`` for one storm.

    Starts when the observation first exceeds ``min_intensity_kts`` and ends
    shortly after decay. Whichever of the observation and the model weakens
    first sets the end, so a storm that the model spins down while the best
    track holds a long weak plateau is not plotted as a mostly-empty axis.
    """
    v = np.asarray(v_obz_kts, dtype=float)
    T = int(v.size)
    valid = np.isfinite(v)
    above_min = valid & (v > float(min_intensity_kts))
    if not above_min.any():
        return 0, T
    i0 = int(np.where(above_min)[0][0])

    obs_last = _last_strong_index(v, i0, min_intensity_kts, peak_drop_frac)
    if obs_last is None:
        return i0, T

    last_strong = obs_last
    if model_kts is not None:
        model_last = _last_strong_index(model_kts, i0, min_intensity_kts, peak_drop_frac)
        if model_last is not None:
            last_strong = min(obs_last, model_last)

    i1 = min(T, last_strong + 1 + int(tail_buffer_hours))
    return i0, max(i1, i0 + 2)


def plot_single_track(result, out_path, min_intensity_kts=30.0,
                      peak_drop_frac=0.4, tail_buffer_hours=24, dpi=150):
    """Plot observed / FAST / FAST_ML intensity with the ventilation panel.

    Args:
        result: Output of :func:`fastml.inference.run_storm`, or a dict read back
            from the single-track NetCDF.
        out_path: Output path; both ``.png`` and ``.svg`` are written.

    Returns:
        The written PNG :class:`~pathlib.Path`.
    """
    import pandas as pd

    n = int(result.get("seq_valid", result["T"]))
    v_obz = np.asarray(result["v_obz_kts"], dtype=float)[:n]
    fast = np.asarray(result["fast_vmax_kts"], dtype=float)[:n]
    ml = np.asarray(result["ml_vmax_kts"], dtype=float)[:n]

    times = result.get("times")
    if times is not None:
        times = pd.to_datetime(np.asarray(times).ravel()[:n])
    else:
        times = pd.date_range("2000-01-01", periods=n, freq="h")

    i0, i1 = find_plot_window(
        v_obz,
        min_intensity_kts=min_intensity_kts,
        peak_drop_frac=peak_drop_frac,
        tail_buffer_hours=tail_buffer_hours,
        model_kts=np.fmax(ml, fast),
    )
    sl = slice(i0, i1)
    t_p = times[sl]

    has_vent = "fast_vent" in result and "ml_vent" in result
    if has_vent:
        fig, (ax, axv) = plt.subplots(
            2, 1, figsize=(14, 7), facecolor="white", sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
        axv = None

    ax.set_facecolor("white")
    ax.plot(t_p, v_obz[sl], label="IBTrACS", color=COLOR_OBS, lw=2.6, alpha=0.95)
    ax.plot(t_p, ml[sl], label="FAST_ML", color=COLOR_FASTML, lw=2.6, alpha=0.95)
    ax.plot(t_p, fast[sl], label="FAST", color=COLOR_FAST, lw=2.6, alpha=0.95)
    ax.set_ylabel("Intensity (knots)", fontsize=22)
    ax.set_ylim(0, 200)
    if len(t_p) > 0:
        ax.set_xlim(pd.Timestamp(t_p.min()), pd.Timestamp(t_p.max()))
    ax.tick_params(axis="both", labelsize=18)

    name = str(result.get("hurricane", "")).split("_")[-1]
    year = result.get("year", "")
    ax.set_title(f"{year} {name}".strip(), fontsize=26, fontweight="bold")
    ax.legend(loc="best", fontsize=20, framealpha=0.9)
    ax.grid(True, alpha=0.3, color="grey")

    if has_vent:
        axv.set_facecolor("white")
        axv.plot(t_p, np.asarray(result["fast_vent"], dtype=float)[:n][sl],
                 color=COLOR_FAST_ENS, lw=3.0, alpha=0.95, label="FAST")
        axv.plot(t_p, np.asarray(result["ml_vent"], dtype=float)[:n][sl],
                 color=COLOR_FASTML_ENS, lw=3.0, alpha=0.95, label="FAST_ML")
        axv.set_ylabel("vent  (chi*S)", fontsize=22)
        axv.set_xlabel("Date", fontsize=22)
        axv.set_ylim(0, None)
        axv.grid(True, alpha=0.3, color="grey")
        axv.tick_params(axis="both", labelsize=18)
        x_axis = axv
    else:
        ax.set_xlabel("Date", fontsize=22)
        x_axis = ax

    x_axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    x_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


# ── Ensemble figure ──────────────────────────────────────────────────────────

def _ensemble_stats(arr, top_frac=0.10):
    """Ensemble mean and the mean of the strongest ``top_frac`` members.

    The top decile matters because intensity ensembles are strongly
    right-skewed: the mean of a large ensemble systematically under-forecasts
    rapid intensification even when some members capture it.
    """
    n = arr.shape[0]
    n_top = max(1, int(n * top_frac))
    with warnings.catch_warnings():
        # Trailing steps past every member's length are legitimately all-NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.argsort(np.nanmax(arr, 1))[-n_top:]
        return np.nanmean(arr, 0), np.nanmean(arr[top], 0)


def plot_ensemble_vs_google(ode_nc, google_csv, out_path, google_id=None,
                            chi_nc=None, storm=None, init_time=None,
                            max_hours=240, google_lead_shift_h=0.0,
                            run_label=None, dpi=170):
    """Plot the FAST / FAST_ML GEFS ensemble against Google FNV3 and IBTrACS.

    Args:
        ode_nc: Ensemble ODE output with ``fast_vmax_kts``, ``ml_vmax_kts`` on
            ``(member, step)``, and the best track embedded as ``v_obz_kts``.
        google_csv: Google WeatherLab FNV3 paired ensemble CSV.
        out_path: Output path; both ``.png`` and ``.svg`` are written.
        google_id: FNV3 ``track_id`` to select, e.g. ``'EP162025'``.
        chi_nc: Optional per-member ``chi``/``S`` file for the ventilation panel.
        storm, init_time: Labels for the title and x-axis.
        max_hours: X-axis cap. Without it the axis stretches over padded steps
            and compresses the actual forecast into a corner.
        google_lead_shift_h: Offset applied to the Google lead time when its
            initialisation differs from ours, in hours.
        run_label: Parenthetical describing our configuration in the title,
            e.g. ``'free run'``. Omitted when ``None``.

    Returns:
        The written PNG :class:`~pathlib.Path`.
    """
    import pandas as pd
    import xarray as xr

    ode_nc = Path(ode_nc)
    if not ode_nc.exists():
        raise FileNotFoundError(
            f"Ensemble ODE file not found: {ode_nc}\n"
            "Run scripts/download_data.py --ensemble to fetch it."
        )

    with xr.open_dataset(ode_nc) as ds:
        fast = ds["fast_vmax_kts"].values
        ml = ds["ml_vmax_kts"].values
        obs = ds["v_obz_kts"].values[0] if "v_obz_kts" in ds else None

    n_obs = int(np.isfinite(obs).sum()) if obs is not None else fast.shape[1]
    hours = np.arange(fast.shape[1])

    # ── Ventilation panel ────────────────────────────────────────────────────
    fast_vent = ml_vent = hours_vent = None
    if chi_nc and Path(chi_nc).exists():
        with xr.open_dataset(chi_nc) as cs:
            seq_len = cs["seq_len"].values if "seq_len" in cs else None
            fast_v = cs["fast_chi"].values * cs["fast_s"].values
            ml_v = cs["ml_chi"].values * cs["ml_s"].values

        # The ventilation archive may hold more members than were successfully
        # integrated. Describe the same ensemble as the panel above, otherwise
        # the two panels would summarise different member sets.
        n_members = fast.shape[0]
        if fast_v.shape[0] > n_members:
            fast_v = fast_v[:n_members]
            ml_v = ml_v[:n_members]
            seq_len = seq_len[:n_members] if seq_len is not None else None

        if seq_len is not None:
            # Blank the edge padding so it cannot bias the percentile band.
            for m in range(fast_v.shape[0]):
                n = int(seq_len[m])
                fast_v[m, n:] = np.nan
                ml_v[m, n:] = np.nan

        with warnings.catch_warnings():
            # Steps beyond every member's length are legitimately all-NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            fast_vent = (np.nanmean(fast_v, 0), np.nanpercentile(fast_v, 10, 0),
                         np.nanpercentile(fast_v, 90, 0))
            ml_vent = (np.nanmean(ml_v, 0), np.nanpercentile(ml_v, 10, 0),
                       np.nanpercentile(ml_v, 90, 0))
        hours_vent = np.arange(fast_v.shape[1])

    # ── Google FNV3 ensemble ─────────────────────────────────────────────────
    google_csv = Path(google_csv)
    if not google_csv.exists():
        raise FileNotFoundError(f"Google FNV3 CSV not found: {google_csv}")
    g = pd.read_csv(google_csv, comment="#")
    if google_id:
        g = g[g["track_id"] == google_id]
    if g.empty:
        raise ValueError(f"No Google rows for track_id={google_id!r} in {google_csv.name}")
    g = g.copy()
    g["lead_h"] = pd.to_timedelta(g["lead_time"]).dt.total_seconds() / 3600.0 + google_lead_shift_h
    g_lead = np.sort(g["lead_h"].unique())
    g_by = {s: grp.set_index("lead_h")["maximum_sustained_wind_speed_knots"]
            for s, grp in g.groupby("sample")}

    g_mat = np.full((len(g_by), len(g_lead)), np.nan)
    for i, ser in enumerate(g_by.values()):
        g_mat[i] = np.interp(g_lead, ser.index.values, ser.values, left=np.nan, right=np.nan)
    g_mean, g_top = _ensemble_stats(g_mat)

    fast_mean, fast_top = _ensemble_stats(fast)
    ml_mean, ml_top = _ensemble_stats(ml)

    # ── Draw ─────────────────────────────────────────────────────────────────
    two_panel = fast_vent is not None
    if two_panel:
        fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 9), dpi=dpi, sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    else:
        fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
        axv = None

    for i in range(fast.shape[0]):
        ax.plot(hours, fast[i], color="#bbf7d0", lw=0.2, alpha=0.22)
    for i in range(ml.shape[0]):
        ax.plot(hours, ml[i], color="#bfdbfe", lw=0.2, alpha=0.22)
    for i in range(g_mat.shape[0]):
        ax.plot(g_lead, g_mat[i], color="#fecaca", lw=0.3, alpha=0.28)

    if obs is not None:
        ax.plot(np.arange(n_obs), obs[:n_obs], "k-", lw=3.0, zorder=9,
                label=f"IBTrACS (peak {np.nanmax(obs[:n_obs]):.0f})")
    ax.plot(hours, ml_mean, color=COLOR_FASTML_ENS, lw=2.4, zorder=8,
            label=f"FAST-ML mean (peak {np.nanmax(ml_mean):.0f})")
    ax.plot(hours, ml_top, color=COLOR_FASTML_ENS, lw=2.0, ls="--", zorder=8,
            label=f"FAST-ML top10% (peak {np.nanmax(ml_top):.0f})")
    ax.plot(hours, fast_mean, color=COLOR_FAST_ENS, lw=2.2, zorder=7,
            label=f"FAST mean (peak {np.nanmax(fast_mean):.0f})")
    ax.plot(g_lead, g_mean, color=COLOR_GOOGLE, lw=2.4, marker="o", ms=4, zorder=8,
            label=f"Google FNV3 mean (peak {np.nanmax(g_mean):.0f}, {len(g_by)} mem)")
    ax.plot(g_lead, g_top, color=COLOR_GOOGLE, lw=2.0, ls="--", marker="s", ms=3.5,
            mfc="none", zorder=8, label=f"Google FNV3 top10% (peak {np.nanmax(g_top):.0f})")

    y_top = (np.nanmax(obs[:n_obs]) if obs is not None else 120) + 30
    ax.set_xlim(0, max_hours if max_hours is not None else max(n_obs, g_lead.max()))
    ax.set_ylim(0, max(160, y_top))
    ax.grid(alpha=0.25)
    ax.set_ylabel("Intensity (kt)")
    title_name = (storm or "").strip().upper()
    ours = "FAST / FAST-ML" + (f" ({run_label})" if run_label else "")
    ax.set_title(
        (f"{title_name}: " if title_name else "")
        + f"Intensity ensemble: {ours} vs Google FNV3 vs IBTrACS",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    x_label = "Hours since init" + (f"   (init: {init_time})" if init_time else "")
    if two_panel:
        axv.fill_between(hours_vent, fast_vent[1], fast_vent[2], color=COLOR_FAST_ENS, alpha=0.15)
        axv.fill_between(hours_vent, ml_vent[1], ml_vent[2], color=COLOR_FASTML_ENS, alpha=0.15)
        axv.plot(hours_vent, fast_vent[0], color=COLOR_FAST_ENS, lw=2.2,
                 label=f"FAST vent=chi*s (mean {np.nanmean(fast_vent[0]):.1f})")
        axv.plot(hours_vent, ml_vent[0], color=COLOR_FASTML_ENS, lw=2.2,
                 label=f"FAST-ML vent=chi*s (mean {np.nanmean(ml_vent[0]):.1f})")
        axv.grid(alpha=0.25)
        axv.set_ylabel("vent  (chi*S)")
        axv.set_xlabel(x_label)
        axv.set_ylim(0, None)
        axv.legend(loc="upper right", fontsize=9, framealpha=0.9)
    else:
        ax.set_xlabel(x_label)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"  FAST-ML mean peak={np.nanmax(ml_mean):.0f} top10%={np.nanmax(ml_top):.0f} | "
          f"Google mean={np.nanmax(g_mean):.0f} top10%={np.nanmax(g_top):.0f}"
          + (f" | obs={np.nanmax(obs[:n_obs]):.0f}" if obs is not None else ""))
    return png
