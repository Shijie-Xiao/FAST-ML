#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants and repository paths.

Every threshold here is the value used to produce the published figures; they
are collected in one module so that FAST and FAST_ML provably run with an
identical configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Repository layout ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "ckpt"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_CKPT = CKPT_DIR / "twostream_final_d2.pth"
#: Statistics over the 2003-2022 training storms, as used at training time and
#: by the published evaluation. Part of the model definition: the training
#: storms are not released, so this file is committed rather than recomputed.
DEFAULT_STATS = DATA_DIR / "spatial_stats_train2003_2022.pkl"
DEFAULT_CD_NC = DATA_DIR / "Cd.nc"

#: Per-storm ERA5 inputs. Too large for git; see ``scripts/download_data.py``.
TRAINING_DATA_DIR = Path(
    os.environ.get("FASTML_TRAINING_DATA", str(DATA_DIR / "training_data"))
)

#: Pre-computed ensemble products for the two GEFS case studies.
ENSEMBLE_DIR = DATA_DIR / "ensemble"


def cd_nc_path() -> Path:
    """Locate ``Cd.nc``, honouring the ``FASTML_CD_NC`` override."""
    override = os.environ.get("FASTML_CD_NC")
    return Path(override) if override else DEFAULT_CD_NC


# ── Unit conversion ──────────────────────────────────────────────────────────
#: All internal computation is in m/s; outputs and figures are in knots.
MS_TO_KNOTS = 1.94384

# ── Spatial input geometry ───────────────────────────────────────────────────
SPATIAL_H, SPATIAL_W = 72, 72
#: 3-D channel order is (T, Q, U, V, Z); 2-D is (SST, MSLP).
N_3D_VARS, N_2D_VARS, N_LEVELS = 5, 2, 7
#: Pressure levels of the 3-D fields, hPa.
LEVELS_HPA = (1000, 850, 700, 600, 500, 400, 300)
#: Sequences are padded to this length so batches are rectangular.
TARGET_SEQ_LEN = 480
#: Encoder mini-batch size, to bound peak GPU memory.
CNN_CHUNK = 32

# ── Forecast configuration ───────────────────────────────────────────────────
#: The forecast starts when the observation first reaches this intensity.
VMAX_START_KTS = 45.0
VMAX_START_MS = VMAX_START_KTS / MS_TO_KNOTS
#: Hours of observation-nudged initialisation before the forecast start.
INIT_HOURS = 48
#: e-folding scale of the inherited-forcing decay, hours.
T0_DECAY_HOURS = 24.0
#: Storms must peak at or above this intensity to be evaluated.
MIN_VMAX_KTS = 45.0
#: Storms must have at least this many hours of forecast after the 45 kt
#: threshold. This is the evaluation threshold used in the manuscript and
#: admits the 12 storms shown in the 2024 test panel of Figure 3.
MIN_DURATION_H = 72

# ── ODE integration ──────────────────────────────────────────────────────────
#: Heun sub-steps per 1 h output step.
SUB_STEPS = 4
STEP_SIZE = 1.0 / SUB_STEPS
#: Floor on ``chi * S``; a hard zero would prevent m from decaying over land.
XS_NAN_FALLBACK = 1e-5

# ── Ventilation calibration ──────────────────────────────────────────────────
#: Mean-to-90th-percentile factor for the dry-air ventilation index.
CHI_MULTIPLIER = 5
#: Physical ceiling on the ventilation index in the Atlantic.
CHI_D_ATLANTIC = 4.0

# ── Boundary layer and drag ──────────────────────────────────────────────────
#: Atmospheric boundary-layer depth for NA/EP, m.
H_BL = 1400.0
#: Neutral over-ocean drag coefficient.
CD_CONST = 1.2e-3
#: Basin bounds as (lon_min, lat_min, lon_max, lat_max), used to clip Cd.nc.
BASIN_BOUNDS = {
    "NA": ["260E", "0N", "360E", "60N"],
    "EP": ["180E", "0N", "290E", "60N"],
    "WP": ["100E", "0N", "180E", "60N"],
}

# ── Figure colours (shared by single-track and ensemble figures) ─────────────
COLOR_OBS = "black"
COLOR_FAST = "#5ec64a"
COLOR_FASTML = "#3a86ff"
COLOR_FAST_ENS = "#16a34a"
COLOR_FASTML_ENS = "#2563eb"
COLOR_GOOGLE = "#dc2626"
