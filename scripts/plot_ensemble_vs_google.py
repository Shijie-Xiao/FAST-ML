#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the two GEFS ensemble case studies against Google FNV3.

Both cases initialise FAST and FAST_ML from a 1000-member GEFS-driven synthetic
track ensemble and compare the resulting intensity distribution with Google
WeatherLab FNV3 and the IBTrACS best track.

The per-member ODE integration and ventilation prediction are already stored in
the distributed NetCDF files, so this script only reads and plots them; it needs
neither a GPU nor the raw GEFS archive.

Examples::

    python scripts/plot_ensemble_vs_google.py            # both cases
    python scripts/plot_ensemble_vs_google.py --case priscilla
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastml.config import ENSEMBLE_DIR, RESULTS_DIR  # noqa: E402
from fastml.plotting import plot_ensemble_vs_google  # noqa: E402

#: Published configuration of each case study.
#:
#: ``run_label`` records whether the inherited-forcing term was retained:
#: Flossie keeps it (the storm was already intensifying at initialisation),
#: while Priscilla is a free run, which isolates the ventilation response.
CASES = {
    "flossie": {
        "ode_nc": "flossie/ode_flossie_midfinit_finit.nc",
        "google_csv": "flossie/FNV3_2025_06_29T12_00_paired.csv",
        "chi_nc": "flossie/chi_s_flossie.nc",
        "google_id": "EP062025",
        "storm": "flossie_midfinit",
        "init_time": "2025-06-29 12:00",
        "out_name": "flossie_midfinit_finit_vs_google",
        "run_label": "inherited forcing",
    },
    "priscilla": {
        "ode_nc": "priscilla/ode_priscilla_vp1p10_free.nc",
        "google_csv": "priscilla/FNV3_2025_10_04T18_00_paired.csv",
        "chi_nc": "priscilla/chi_s_priscilla.nc",
        "google_id": "EP162025",
        "storm": "priscilla",
        "init_time": "2025-10-04 18:00",
        "out_name": "priscilla_free_vs_google",
        "run_label": "free run",
    },
    "beryl": {
        "ode_nc": "beryl/ode_beryl_vp1p10_free.nc",
        "google_csv": "beryl/FNV3_2024_06_29T00_00_paired.csv",
        "chi_nc": "beryl/chi_s_beryl.nc",
        "google_id": "AL022024",
        "storm": "beryl",
        "init_time": "2024-06-29 00:00",
        "out_name": "beryl_free_vs_google",
        "run_label": "free run",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the GEFS ensemble vs Google FNV3 figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", nargs="+", choices=sorted(CASES) + ["all"],
                        default=["all"], help="Case studies to plot")
    parser.add_argument("--data-dir", type=Path, default=ENSEMBLE_DIR,
                        help="Root of the ensemble NetCDF and Google CSV files")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "ensemble",
                        help="Output directory for the figures")
    parser.add_argument("--max-hours", type=float, default=240,
                        help="X-axis cap, hours since initialisation")
    parser.add_argument("--no-vent-panel", action="store_true",
                        help="Skip the ventilation panel (avoids the large chi_s files)")
    args = parser.parse_args()

    names = sorted(CASES) if "all" in args.case else args.case
    failures = []

    for name in names:
        case = CASES[name]
        print(f"\n=== {name}  ({case['run_label']}, init {case['init_time']}) ===")

        chi_nc = None
        if not args.no_vent_panel:
            candidate = args.data_dir / case["chi_nc"]
            if candidate.exists():
                chi_nc = candidate
            else:
                print(f"  note: {candidate.name} absent, omitting the ventilation panel.")
                print(f"        Fetch it with: python scripts/download_data.py --ensemble")

        try:
            png = plot_ensemble_vs_google(
                ode_nc=args.data_dir / case["ode_nc"],
                google_csv=args.data_dir / case["google_csv"],
                out_path=args.out_dir / case["out_name"],
                google_id=case["google_id"],
                chi_nc=chi_nc,
                storm=case["storm"],
                init_time=case["init_time"],
                max_hours=args.max_hours,
                run_label=case["run_label"],
            )
            print(f"  wrote {png}")
            print(f"  wrote {png.with_suffix('.svg')}")
        except FileNotFoundError as exc:
            print(f"  FAILED: {exc}")
            failures.append(name)

    if failures:
        print(f"\nIncomplete: {', '.join(failures)}. "
              f"Run scripts/download_data.py --ensemble to fetch the missing inputs.")
        return 1
    print(f"\nFigures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
