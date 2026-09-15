#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-track intensity hindcast: FAST vs FAST_ML.

Runs both configurations over every matching storm, writes one NetCDF and one
figure per storm, and a CSV of the metrics. Inference only -- no training.

Examples::

    # Every 2024 North Atlantic storm (the published demo case).
    python scripts/run_single_track.py --years 2024 --basin NA

    # One storm, on CPU.
    python scripts/run_single_track.py --years 2024 --storm MILTON --device cpu

    # Redraw figures from the NetCDF files, without the model or the ERA5 fields.
    python scripts/run_single_track.py --replot-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastml.config import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_STATS,
    MIN_DURATION_H,
    MIN_VMAX_KTS,
    RESULTS_DIR,
    TRAINING_DATA_DIR,
)
from fastml.data import (  # noqa: E402
    find_storm_dirs,
    load_spatial_stats,
    load_storm,
)
from fastml.inference import run_storm  # noqa: E402
from fastml.model import load_model  # noqa: E402
from fastml.netcdf_io import read_single_track, write_single_track  # noqa: E402
from fastml.plotting import plot_single_track  # noqa: E402

#: Metric columns written to the summary CSV.
_CSV_FIELDS = [
    "storm_id", "hurricane", "year", "n_valid", "t_start",
    "peak_obs_kts", "peak_fast_kts", "peak_ml_kts",
    "fast_rmse_kts", "ml_rmse_kts", "rmse_gain_kts",
    "fast_bias_kts", "ml_bias_kts",
    "fast_corr", "ml_corr",
]


def _replot_from_netcdf(nc_dir, fig_dir):
    """Regenerate figures from existing NetCDF output.

    Demonstrates that the published product files are self-sufficient: the
    figures come back without the checkpoint, a GPU, or the ERA5 fields.
    """
    nc_files = sorted(Path(nc_dir).glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files in {nc_dir}; run inference first.")

    for nc_path in nc_files:
        with read_single_track(nc_path) as ds:
            result = {name: ds[name].values for name in ds.data_vars}
            result["times"] = ds["time"].values
            result["T"] = ds.sizes["time"]
            result["seq_valid"] = ds.sizes["time"]
            result["lats"] = result.pop("lat")
            result["lons"] = result.pop("lon")
            result["hurricane"] = ds.attrs.get("hurricane", nc_path.stem)
            result["year"] = int(ds.attrs.get("year", 0))
        png = plot_single_track(result, Path(fig_dir) / nc_path.stem)
        print(f"  {nc_path.name} -> {png.name}")
    print(f"\nRedrew {len(nc_files)} figures from NetCDF into {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="FAST vs FAST_ML single-track hindcast",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR,
                        help="Root of the per-storm ERA5 inputs")
    parser.add_argument("--years", type=int, nargs="+", default=[2024],
                        help="Years to evaluate")
    parser.add_argument("--basin", default="NA", choices=["NA", "EP", "all"],
                        help="Basin filter")
    parser.add_argument("--storm", nargs="+", default=None,
                        help="Restrict to storms whose id contains these names")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="Released model weights")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS,
                        help="Spatial normalisation statistics")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "single_track",
                        help="Output root for NetCDF, figures and metrics")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Torch device for the CNN")
    parser.add_argument("--min-vmax-kts", type=float, default=MIN_VMAX_KTS,
                        help="Skip storms peaking below this intensity")
    parser.add_argument("--min-duration-h", type=int, default=MIN_DURATION_H,
                        help="Skip storms with less forecast time than this")
    parser.add_argument("--no-plot", action="store_true", help="Write NetCDF only")
    parser.add_argument("--replot-only", action="store_true",
                        help="Redraw figures from existing NetCDF and exit")
    args = parser.parse_args()

    nc_dir = args.out_dir / "netcdf"
    fig_dir = args.out_dir / "figures"

    if args.replot_only:
        _replot_from_netcdf(nc_dir, fig_dir)
        return 0

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    storm_dirs = find_storm_dirs(
        args.data_dir,
        years=args.years,
        basin=None if args.basin == "all" else args.basin,
        storm_names=args.storm,
    )
    if not storm_dirs:
        print(f"No storms matched under {args.data_dir} "
              f"(years={args.years}, basin={args.basin}, storm={args.storm})")
        return 1

    print(f"FAST vs FAST_ML single-track hindcast")
    print(f"  data      : {args.data_dir}")
    print(f"  years     : {args.years}   basin: {args.basin}")
    print(f"  candidates: {len(storm_dirs)} storms")
    print(f"  device    : {device}")

    stats = load_spatial_stats(args.stats)
    model = load_model(args.ckpt, device=device)
    print(f"  weights   : {args.ckpt.name} (loaded strict)\n")

    rows, skipped = [], []
    for i, storm_dir in enumerate(storm_dirs, 1):
        storm, reason = load_storm(
            storm_dir,
            min_vmax_kts=args.min_vmax_kts,
            min_duration_h=args.min_duration_h,
        )
        if storm is None:
            skipped.append((storm_dir.name, reason))
            print(f"[{i}/{len(storm_dirs)}] skip {storm_dir.name}: {reason}")
            continue

        t0 = time.time()
        result = run_storm(model, storm, stats, device=device)
        nc_path = write_single_track(result, nc_dir / f"{storm['storm_id']}.nc")
        if not args.no_plot:
            plot_single_track(result, fig_dir / storm["storm_id"])

        rows.append({k: result.get(k) for k in _CSV_FIELDS})
        print(
            f"[{i}/{len(storm_dirs)}] {storm['storm_id']}  "
            f"obs_peak={result['peak_obs_kts']:5.1f}  "
            f"FAST={result['fast_rmse_kts']:5.2f}  "
            f"FAST_ML={result['ml_rmse_kts']:5.2f}  "
            f"gain={result['rmse_gain_kts']:+5.2f} kt  "
            f"({time.time() - t0:.1f}s)"
        )

    if not rows:
        print("\nNo storms produced results.")
        return 1

    metrics_csv = args.out_dir / "storm_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    fast_rmse = np.array([r["fast_rmse_kts"] for r in rows], dtype=float)
    ml_rmse = np.array([r["ml_rmse_kts"] for r in rows], dtype=float)
    ok = np.isfinite(fast_rmse) & np.isfinite(ml_rmse)
    n_better = int((ml_rmse[ok] < fast_rmse[ok]).sum())

    print(f"\n{'=' * 62}")
    print(f"Storms evaluated          : {len(rows)}  ({len(skipped)} skipped)")
    print(f"Mean RMSE, FAST           : {np.nanmean(fast_rmse):.2f} kt")
    print(f"Mean RMSE, FAST_ML        : {np.nanmean(ml_rmse):.2f} kt")
    print(f"Mean improvement          : {np.nanmean(fast_rmse - ml_rmse):+.2f} kt")
    print(f"Storms where FAST_ML wins : {n_better}/{int(ok.sum())}")
    print(f"{'=' * 62}")
    print(f"NetCDF : {nc_dir}")
    if not args.no_plot:
        print(f"Figures: {fig_dir}")
    print(f"Metrics: {metrics_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
