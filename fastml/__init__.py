#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAST_ML -- machine-learned ventilation for the FAST tropical cyclone intensity model.

Reproduction code accompanying the FAST_ML manuscript. The package exposes an
inference-only path: the released CNN diagnoses the ventilation index
``chi * S``, which is then integrated by the same FAST ODE used by the
ERA5-driven baseline.

Typical use::

    from fastml import load_model, load_spatial_stats, load_storm, run_storm

    model = load_model("ckpt/twostream_final_d2.pth", device="cuda")
    stats = load_spatial_stats("data/spatial_stats_2003_2021.pkl")
    storm, _ = load_storm("data/training_data/2024/AL782024_north_atlantic_MILTON")
    result = run_storm(model, storm, stats, device="cuda")
"""

from .config import MS_TO_KNOTS
from .data import (
    find_storm_dirs,
    load_spatial,
    load_spatial_stats,
    load_storm,
    load_storms,
    normalize_spatial,
)
from .fast_physics import chi_calibrated_multiply, run_fast_with_init
from .inference import predict_chi_s, run_storm
from .model import TwoStreamFASTModel, load_model
from .netcdf_io import read_single_track, write_single_track
from .plotting import plot_ensemble_vs_google, plot_single_track

__version__ = "1.0.0"

__all__ = [
    "MS_TO_KNOTS",
    "TwoStreamFASTModel",
    "chi_calibrated_multiply",
    "find_storm_dirs",
    "load_model",
    "load_spatial",
    "load_spatial_stats",
    "load_storm",
    "load_storms",
    "normalize_spatial",
    "plot_ensemble_vs_google",
    "plot_single_track",
    "predict_chi_s",
    "read_single_track",
    "run_fast_with_init",
    "run_storm",
    "write_single_track",
]
