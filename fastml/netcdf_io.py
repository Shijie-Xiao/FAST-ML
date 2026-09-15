#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write and read the single-track NetCDF product.

One file is written per storm. It carries every series the comparison figure
draws plus the ventilation diagnosis behind it, so a figure can be regenerated
from the NetCDF alone -- no model, no GPU, and no multi-gigabyte ERA5 fields.
The metrics are stored as global attributes so the file is self-describing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: (variable name, result key, units, long name).
_VARIABLES = [
    ("v_obz_kts", "v_obz_kts", "knots", "Observed maximum sustained wind (IBTrACS best track)"),
    ("fast_vmax_kts", "fast_vmax_kts", "knots", "FAST maximum sustained wind (ERA5 ventilation)"),
    ("ml_vmax_kts", "ml_vmax_kts", "knots", "FAST_ML maximum sustained wind (CNN ventilation)"),
    ("fast_v_axisym_kts", "fast_v_axisym_kts", "knots", "FAST axisymmetric wind (ODE state variable)"),
    ("ml_v_axisym_kts", "ml_v_axisym_kts", "knots", "FAST_ML axisymmetric wind (ODE state variable)"),
    ("vp_kts", "vp_kts", "knots", "Potential intensity, 3 h median filtered"),
    ("fast_chi", "fast_chi", "1", "FAST calibrated entropy deficit chi from ERA5"),
    ("fast_s", "fast_s", "1", "FAST ventilation shear term S from ERA5"),
    ("fast_vent", "fast_vent", "1", "FAST ventilation index chi*S from ERA5"),
    ("ml_chi", "ml_chi", "1", "FAST_ML predicted entropy deficit chi"),
    ("ml_s", "ml_s", "1", "FAST_ML predicted ventilation shear term S"),
    ("ml_vent", "ml_vent", "1", "FAST_ML ventilation index chi*S"),
    ("fast_m", "fast_m", "1", "FAST moisture state variable m"),
    ("ml_m", "ml_m", "1", "FAST_ML moisture state variable m"),
    ("lat", "lats", "degrees_north", "Storm centre latitude"),
    ("lon", "lons", "degrees_east", "Storm centre longitude"),
]

#: Scalar metrics promoted to global attributes.
_ATTRS = [
    "storm_id", "hurricane", "year", "seq_valid", "t_start",
    "fast_rmse_kts", "fast_bias_kts", "fast_corr",
    "ml_rmse_kts", "ml_bias_kts", "ml_corr", "rmse_gain_kts",
    "n_valid", "peak_obs_kts", "peak_fast_kts", "peak_ml_kts",
]


def write_single_track(result, out_path):
    """Write one storm's FAST vs FAST_ML comparison to NetCDF.

    Only the valid portion of the sequence is written; the edge padding that
    makes batches rectangular is discarded here so the file matches the storm's
    real duration.

    Args:
        result: Output of :func:`fastml.inference.run_storm`.
        out_path: Destination ``.nc`` path; parents are created.

    Returns:
        The written :class:`~pathlib.Path`.
    """
    import pandas as pd
    import xarray as xr

    n = int(result["seq_valid"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times = result.get("times")
    if times is not None:
        time_index = pd.to_datetime(np.asarray(times).ravel()[:n])
    else:
        time_index = pd.date_range("2000-01-01", periods=n, freq="h")

    data_vars = {}
    for name, key, units, long_name in _VARIABLES:
        if key not in result:
            continue
        values = np.asarray(result[key], dtype=np.float64).reshape(-1)[:n]
        if values.size < n:
            values = np.concatenate([values, np.full(n - values.size, np.nan)])
        data_vars[name] = (
            "time",
            values.astype(np.float32),
            {"units": units, "long_name": long_name},
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": ("time", time_index),
            "lead_hours": ("time", np.arange(n, dtype=np.int32)),
        },
    )
    # No 'units' attribute: "hours" would make readers decode this as a
    # timedelta rather than the plain step index it is.
    ds["lead_hours"].attrs = {
        "long_name": "Hours since the start of the 48 h initialisation window",
    }

    for key in _ATTRS:
        if key in result and result[key] is not None:
            value = result[key]
            ds.attrs[key] = value if isinstance(value, str) else float(value)

    ds.attrs.update({
        "title": f"FAST vs FAST_ML intensity forecast: {result.get('hurricane', '')}",
        "summary": (
            "Single-track intensity hindcast. FAST and FAST_ML share one ODE, "
            "one track and one set of scalars; they differ only in the source "
            "of the ventilation index chi*S (ERA5 diagnostics vs the two-stream CNN)."
        ),
        "forecast_start_index": int(result.get("t_start", 48)),
        "forecast_start_note": (
            "Steps before forecast_start_index are the 48 h observation-nudged "
            "initialisation, not a forecast."
        ),
        "source": "https://github.com/Shijie-Xiao/FAST_ML",
    })

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    ds.close()
    return out_path


def read_single_track(nc_path):
    """Open a single-track NetCDF written by :func:`write_single_track`."""
    import xarray as xr

    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")
    return xr.open_dataset(nc_path)
