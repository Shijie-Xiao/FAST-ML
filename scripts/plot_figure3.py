#!/usr/bin/env python3
"""Redraw manuscript Figure 3 from the archived per-storm metrics.

Panel A: per-year mean RMSE, 2003-2024, with the validation and test years
marked. Panels B, C, D: per-storm RMSE for 2022, 2023 and 2024. FAST is green
(#5ec64a), FAST-ML blue (#3a86ff), and the RMSE reduction is annotated above
each pair.

The numbers come from ``results/storm_metrics_all_years_NA.csv``, which scores
all 224 North Atlantic storms with the released inference path. Producing that
CSV needs the full ERA5 archive and so is not part of this release, but every
value drawn here is read straight from it.

Two output modes:

``--composed`` (default)
    One four-panel figure, written as PNG, PDF and SVG. This is the version
    embedded in the README.

``--panels``
    The four panels as separate SVGs, each sized and scaled as in the
    manuscript, for composition in a vector editor.

Usage::

    python scripts/plot_figure3.py                 # img/Figure_3.{png,pdf,svg}
    python scripts/plot_figure3.py --panels        # img/panels/panel*.svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastml.config import REPO_ROOT, RESULTS_DIR  # noqa: E402

C_FAST = "#5ec64a"
C_ML = "#3a86ff"
VAL_YEARS = (2022, 2023)
TEST_YEARS = (2024,)
PANEL_YEARS = ((2022, "Validation", "B"), (2023, "Validation", "C"),
               (2024, "Test", "D"))

DEFAULT_CSV = RESULTS_DIR / "storm_metrics_all_years_NA.csv"
IMG_DIR = REPO_ROOT / "img"


def _yearly_means(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("year")
              .agg(fast=("fast_rmse_kts", "mean"), ml=("ml_rmse_kts", "mean"),
                   n=("hurricane", "count"))
              .sort_index())


def _storms_of(df: pd.DataFrame, year: int) -> pd.DataFrame:
    d = df[df.year == year].copy()
    d["sname"] = d.hurricane.str.split("_").str[-1]
    return d.sort_values("sname").reset_index(drop=True)


# ── composed figure ──────────────────────────────────────────────────────────

def _panel_yearly(ax, df):
    grp = _yearly_means(df)
    years = grp.index.tolist()
    x = np.arange(len(years))
    w = 0.4

    ax.bar(x - w / 2, grp["fast"], w, label="FAST", color=C_FAST,
           edgecolor="black", linewidth=0.6, zorder=2)
    ax.bar(x + w / 2, grp["ml"], w, label="FAST_ML", color=C_ML,
           edgecolor="black", linewidth=0.6, zorder=2)

    ymax = max(grp["fast"].max(), grp["ml"].max())
    y_top = ymax * 1.20
    for i, yr in enumerate(years):
        gain = grp.loc[yr, "fast"] - grp.loc[yr, "ml"]
        top = max(grp.loc[yr, "fast"], grp.loc[yr, "ml"])
        ax.text(i, top + y_top * 0.012, f"{gain:+.1f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="#1a7f37" if gain >= 0 else "#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years], rotation=45, fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Per-storm Mean RMSE (knots)", fontsize=10)
    ax.set_title("Dataset Partitioning (2003-2024)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)

    idx = {yr: i for i, yr in enumerate(years)}
    for label, group in (("Val", VAL_YEARS), ("Test", TEST_YEARS)):
        present = [idx[y] for y in group if y in idx]
        if not present:
            continue
        ax.axvline(min(present) - 0.5, color="#555555", ls="--", lw=1.2,
                   alpha=0.85, zorder=3)
        ax.text((min(present) + max(present)) / 2, y_top * 0.985, label,
                ha="center", va="top", fontsize=9.5, fontweight="bold", zorder=5)
    return grp


def _panel_storms(ax, df, year, split):
    d = _storms_of(df, year)
    n = len(d)
    x = np.arange(n)
    w = 0.42

    ax.bar(x - w / 2, d.fast_rmse_kts, w, label="FAST", color=C_FAST,
           edgecolor="black", linewidth=0.5, zorder=2)
    ax.bar(x + w / 2, d.ml_rmse_kts, w, label="FAST_ML", color=C_ML,
           edgecolor="black", linewidth=0.5, zorder=2)

    ymax = max(d.fast_rmse_kts.max(), d.ml_rmse_kts.max())
    y_top = ymax * 1.22
    for i in range(n):
        gain = d.fast_rmse_kts[i] - d.ml_rmse_kts[i]
        top = max(d.fast_rmse_kts[i], d.ml_rmse_kts[i])
        ax.text(i, top + y_top * 0.012, f"{gain:+.1f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="#1a7f37" if gain >= 0 else "#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels(d.sname, rotation=90, fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Per-storm RMSE (knots)", fontsize=10)
    ax.set_title(f"{year}  ({split})", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    return d


def write_composed(df: pd.DataFrame, out: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(14.5, 7.6), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.17,
                  left=0.055, right=0.985, top=0.93, bottom=0.09)
    axA, axB, axC, axD = (fig.add_subplot(gs[i, j])
                          for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    for ax in (axA, axB, axC, axD):
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_linewidth(1.1)

    _panel_yearly(axA, df)
    for ax, (year, split, _) in zip((axB, axC, axD), PANEL_YEARS):
        _panel_storms(ax, df, year, split)
    axC.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="black")

    for ax, lab in ((axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")):
        ax.text(-0.075, 1.16, lab, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="top", ha="left")

    written = []
    for suffix in (".png", ".pdf", ".svg"):
        p = out.with_suffix(suffix)
        fig.savefig(p, dpi=dpi if suffix == ".png" else None,
                    bbox_inches="tight", facecolor="white")
        written.append(p)
    plt.close(fig)
    return written


# ── separate panels, manuscript scaling ──────────────────────────────────────

def write_panel_yearly(df: pd.DataFrame, out: Path) -> Path:
    grp = _yearly_means(df)
    years = grp.index.tolist()
    x = np.arange(len(years))
    w = 0.4

    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(x - w / 2, grp["fast"], w, label="FAST", color=C_FAST,
           edgecolor="black", linewidth=0.6, zorder=2)
    ax.bar(x + w / 2, grp["ml"], w, label="FAST_ML", color=C_ML,
           edgecolor="black", linewidth=0.6, zorder=2)

    for i, yr in enumerate(years):
        gain = grp.loc[yr, "fast"] - grp.loc[yr, "ml"]
        top = max(grp.loc[yr, "fast"], grp.loc[yr, "ml"])
        ax.text(i, top + 0.5, f"{gain:+.1f}", ha="center", va="bottom",
                color="#222222", fontsize=16, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years], rotation=45, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_xlabel("Year", fontsize=22)
    ax.set_ylabel("Per-storm Mean RMSE (knots)", fontsize=22)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.legend(loc="upper left", fontsize=22, framealpha=0.95, edgecolor="black")
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(years) - 0.4)
    ymax = max(grp["fast"].max(), grp["ml"].max())
    y_top = ymax * 1.18
    ax.set_ylim(0, y_top)

    idx = {int(y): i for i, y in enumerate(years)}
    for label, group in (("Val", VAL_YEARS), ("Test", TEST_YEARS)):
        present = [idx[y] for y in group if y in idx]
        if not present:
            continue
        ax.axvline(min(present) - 0.5, color="#555555", linestyle="--",
                   linewidth=1.8, alpha=0.8, zorder=3)
        ax.text((min(present) + max(present)) / 2, y_top * 0.99, label,
                ha="center", va="top", fontsize=21, fontweight="bold",
                color="black", zorder=5)

    out = out.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_panel_storms(df: pd.DataFrame, year: int, split: str, out: Path) -> Path:
    d = _storms_of(df, year)
    n = len(d)
    x = np.arange(n)
    w = 0.42

    fig, ax = plt.subplots(figsize=(max(9, n * 0.62), 6.8), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(x - w / 2, d.fast_rmse_kts, w, label="FAST", color=C_FAST,
           edgecolor="black", linewidth=0.5, zorder=2)
    ax.bar(x + w / 2, d.ml_rmse_kts, w, label="FAST_ML", color=C_ML,
           edgecolor="black", linewidth=0.5, zorder=2)

    ymax = max(d.fast_rmse_kts.max(), d.ml_rmse_kts.max())
    y_top = ymax * 1.20
    for i in range(n):
        gain = d.fast_rmse_kts[i] - d.ml_rmse_kts[i]
        top = max(d.fast_rmse_kts[i], d.ml_rmse_kts[i])
        ax.text(i, top + y_top * 0.012, f"{gain:+.1f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                color="#1a7f37" if gain >= 0 else "#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels(d.sname, rotation=90, fontsize=14)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Per-storm RMSE (knots)", fontsize=22)
    ax.set_title(f"{year}  ({split})", fontsize=24, fontweight="bold")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.legend(loc="upper left", fontsize=20, framealpha=0.95, edgecolor="black")

    out = out.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--panels", action="store_true",
                    help="write the four panels as separate SVGs")
    ap.add_argument("--out", type=Path, default=None,
                    help="output stem (composed) or directory (--panels)")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["fast_rmse_kts", "ml_rmse_kts"])
    print(f"{len(df)} North Atlantic storms, {int(df.year.min())}-{int(df.year.max())}")

    if args.panels:
        out_dir = args.out or (IMG_DIR / "panels")
        out_dir.mkdir(parents=True, exist_ok=True)
        written = [write_panel_yearly(df, out_dir / "panelA_yearly_rmse")]
        for year, split, panel in PANEL_YEARS:
            written.append(write_panel_storms(
                df, year, split, out_dir / f"panel{panel}_storm_rmse_{year}"))
    else:
        out = args.out or (IMG_DIR / "Figure_3")
        out.parent.mkdir(parents=True, exist_ok=True)
        written = write_composed(df, out, args.dpi)

    for p in written:
        print(f"  wrote {p.relative_to(REPO_ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")

    grp = _yearly_means(df)
    grp["gain"] = grp.fast - grp.ml
    print("\nper-year means:")
    print(grp.to_string(float_format=lambda v: f"{v:7.2f}"))
    print(f"\nall years: FAST {df.fast_rmse_kts.mean():.2f}, "
          f"FAST-ML {df.ml_rmse_kts.mean():.2f}, "
          f"gain {df.rmse_gain_kts.mean():+.2f}, "
          f"improved {int((df.rmse_gain_kts > 0).sum())}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
