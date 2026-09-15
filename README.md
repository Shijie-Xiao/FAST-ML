# FAST_ML

Machine-learned ventilation for the FAST tropical cyclone intensity model.

FAST predicts tropical cyclone intensity by integrating a coupled ODE for the
axisymmetric wind `V` and a moisture variable `m`. Its dominant source of
uncertainty is the **ventilation index** `chi * S` -- the rate at which low
entropy environmental air is stirred into the core -- which is conventionally
diagnosed from reanalysis. FAST_ML replaces that diagnosis with a two-stream
CNN applied to storm-centred ERA5 fields, and changes nothing else.

That isolation is the point of this repository: FAST and FAST_ML share one ODE,
one track, one set of scalars, one drag field and one initialisation procedure.
The only thing that differs is where `chi * S` comes from, so any difference in
skill is attributable to the ventilation diagnosis alone.

```
        ERA5 chi_ref, s_ref  ──┐
                               ├──>  chi * S  ──>  FAST ODE  ──>  intensity
   72x72 fields ──> CNN ──────┘                (identical for both paths)
```

This is an **inference-only** release. It reproduces the published results; it
does not contain the training loop or the code that derives the ERA5 ventilation
diagnostics from raw reanalysis.

## What you can reproduce

| Experiment | Command | Download needed |
|---|---|---|
| Figures from the archived NetCDF -- no model, no GPU | `run_single_track.py --replot-only` | none |
| GEFS ensemble case studies vs Google FNV3 | `plot_ensemble_vs_google.py` | 53 MB |
| Single-track hindcast, all 2024 North Atlantic storms | `run_single_track.py --years 2024 --basin NA` | 2.1 GB |

The first row is the reproducibility guarantee: every published figure can be
regenerated from the NetCDF files committed under `results/`, which are
themselves written by the third row. Nothing in the figure pipeline depends on
state outside this repository.

Single-track results over the 2024 North Atlantic season, RMSE against IBTrACS
in knots over the full forecast period:

| Storm | Observed peak | FAST | FAST_ML | Improvement |
|---|---:|---:|---:|---:|
| BERYL | 145 | 12.78 | 10.35 | +2.42 |
| DEBBY | 70 | 18.60 | 16.32 | +2.28 |
| ERNESTO | 85 | 10.31 | 10.68 | -0.37 |
| ISAAC | 90 | 28.97 | 9.17 | +19.80 |
| KIRK | 130 | 20.59 | 14.90 | +5.69 |
| LESLIE | 90 | 21.78 | 14.11 | +7.67 |
| MILTON | 155 | 29.24 | 19.81 | +9.43 |
| RAFAEL | 105 | 17.20 | 12.52 | +4.68 |
| **mean** | | **19.93** | **13.48** | **+6.45** |

The per-storm numbers are in `results/single_track/storm_metrics.csv`.

## Install

```bash
conda env create -f environment.yml
conda activate fast_ml
```

Or into an existing environment with Python >= 3.9:

```bash
pip install -r requirements.txt
```

A GPU is optional. The CNN runs on CPU at roughly 1-2 minutes per storm instead
of a few seconds; pass `--device cpu`.

## Quick start

Pick the level of reproduction you need. Each level is self-contained.

### Level 1 -- figures only, nothing to download (about 1 minute)

Everything required is already in the repository.

```bash
git clone https://github.com/Shijie-Xiao/FAST_ML.git
cd FAST_ML
pip install -r requirements.txt

# Redraw every single-track figure from the archived NetCDF.
python scripts/run_single_track.py --replot-only

# Redraw the ensemble figures (intensity panel only, no download needed).
python scripts/plot_ensemble_vs_google.py --no-vent-panel
```

The regenerated figures match the ones committed under `results/`. On the same
machine they come out byte-identical; across machines the anti-aliasing in
matplotlib's rasteriser can differ by one greyscale level on a few dozen of the
2.2 million pixels, which is invisible. The numbers behind the figures are not
approximate: see the note on exactness below.

### Level 2 -- the published ensemble figures (about 53 MB, 2 minutes)

```bash
python scripts/download_data.py --ensemble
python scripts/plot_ensemble_vs_google.py
```

This reproduces both case-study figures in full, including the ventilation
panels.

### Level 3 -- run the model end to end (about 2.1 GB, 5 minutes on a GPU)

```bash
python scripts/download_data.py --single-track
python scripts/run_single_track.py --years 2024 --basin NA
```

This runs the CNN and the ODE over the 2024 North Atlantic season and rewrites
the NetCDF, figures and metrics from scratch. Expected output:

```
Storms evaluated          : 8  (11 skipped)
Mean RMSE, FAST           : 19.93 kt
Mean RMSE, FAST_ML        : 13.48 kt
Mean improvement          : +6.45 kt
Storms where FAST_ML wins : 7/8
```

Eleven of the nineteen storms are skipped by design, and the run reports why
for each: they either never reach the 45 kt forecast threshold or have under
120 h of forecast after it.

To check which inputs are present without downloading anything:

```bash
python scripts/download_data.py --verify
```

### A single storm

```bash
python scripts/run_single_track.py --years 2024 --storm MILTON
```

```
[1/1] AL782024_north_atlantic_MILTON  obs_peak=155.0  FAST=29.24  FAST_ML=19.81  gain=+9.43 kt
```

`gain` is the RMSE reduction of FAST_ML relative to FAST, in knots, scored
against the IBTrACS best track over the whole forecast period.

## Repository layout

```
fastml/
  config.py        Constants and thresholds, in one place, shared by both paths
  model.py         Two-stream CNN that diagnoses chi and S  (inference only)
  fast_physics.py  The FAST ODE, 48 h initialisation, wind conversion (NumPy)
  data.py          Storm loading, time alignment, normalisation
  inference.py     Runs FAST and FAST_ML over one storm
  netcdf_io.py     The single-track NetCDF product
  plotting.py      Single-track and ensemble figures
scripts/
  run_single_track.py        Single-track hindcast driver
  plot_ensemble_vs_google.py Ensemble case-study figures
  compute_spatial_stats.py   Regenerates the normalisation statistics
  download_data.py           Fetches the files hosted outside git
ckpt/
  twostream_final_d2.pth     Released weights (4.7 MB, 237 tensors)
data/
  spatial_stats_train2003_2022.pkl  Normalisation statistics
  Cd.nc                             Spatially varying drag coefficient
  ensemble/<case>/                  Ensemble ODE output + Google FNV3 reference
  manifest.json                     Locations and checksums of the large files
results/
  single_track/netcdf/  Published per-storm output
  single_track/figures/ Published per-storm figures
  ensemble/             Published ensemble figures
```

## Method summary

**Time axis.** The forecast starts when the best track first reaches 45 kt. The
preceding 48 h initialise the ODE, so every sequence is cut to begin 48 h before
that threshold:

```
t=0                      t_start (=48)                    end
 |<- 48 h initialisation ->|<--------- forecast --------->|
```

**Initialisation.** Over the 48 h window, `V` is nudged onto the observation and
the residual `F = observed acceleration - physics RHS` is recorded, while `m`
spins up freely. From `t_start` the integration is free, with the inherited
forcing decaying as `F * exp(-2 * (lead / 24 h)^2)`. Without this the forecast
would discard the momentum the storm already had at initialisation.

**Observation inversion.** The best track reports the maximum sustained wind,
which contains the storm translation and the shear-induced asymmetry, whereas
the ODE state is the axisymmetric mean. Observations are therefore inverted to
axisymmetric form before being used as a target, and model output is converted
back for scoring.

**Ventilation calibration.** The relevant quantity is the driest air reaching
the core, not the annulus mean. Both paths apply the same mean-to-90th-percentile
factor and the same Atlantic ceiling, so the ERA5 and CNN ventilation indices
live on one common scale.

**The two streams.** `S` is predicted from the wind and geopotential fields
(U, V, Z) and `chi` from the thermodynamic fields (T, Q, SST, MSLP). The split
follows the physics: `S` is a shear quantity and `chi` an entropy deficit.
Neither stream is recurrent -- both are per-timestep diagnostics -- which keeps
the network from learning the observed intensity trajectory and forces all
temporal evolution to come from the ODE.

## How exact is the reproduction

Verified by cloning this repository into an empty directory, fetching the hosted
files, and rerunning everything:

| Output | Agreement with the committed results |
|---|---|
| `results/single_track/netcdf/*.nc` | bit-identical, all 8 storms, every variable |
| `results/single_track/storm_metrics.csv` | identical |
| Ensemble peak and ventilation values in the legends | identical |
| Figures | visually identical; see below |

The NetCDF output and every number quoted in the manuscript are exact. The PNGs
are byte-identical when regenerated on the same machine, but matplotlib's
anti-aliasing is not bit-reproducible across machines: a fresh clone reproduced
3 of the 8 single-track figures byte-for-byte and the other 5 differed by one
greyscale level on 6 to 31 pixels out of 2.2 million. If you need to compare
figures mechanically, compare the NetCDF instead.

Two deliberate numerical choices are worth recording, because they make this
code differ in the last digits from the internal version that produced the
manuscript:

- Track coordinates are kept in float64 throughout. The internal pipeline
  briefly cast them to float32, which perturbs the Coriolis parameter and drag
  lookup at the 1e-4 kt level in the final intensity. This version is the more
  accurate of the two.
- The `vp_kts` variable stored in the NetCDF is the median-filtered series that
  the ODE is actually integrated with. The internal version reported the
  unfiltered series alongside a filtered integration, which was misleading.

Normalisation statistics are the one thing that cannot be derived from what is
distributed here, because they come from the training set. They are therefore
committed directly as `data/spatial_stats_train2003_2022.pkl` (under 1 KB), and
`scripts/compute_spatial_stats.py` documents and regenerates them if you have
the training archive.

## Data distribution

Everything needed to **redraw** the published figures is in git (about 20 MB).
Only running the CNN, or drawing the ensemble ventilation panels, needs the
externally hosted files.

**In this git repository:**

| File | Size | Role |
|---|---|---|
| `ckpt/twostream_final_d2.pth` | 4.7 MB | released model weights |
| `data/Cd.nc` | 8.3 MB | spatially varying drag coefficient |
| `data/spatial_stats_train2003_2022.pkl` | <1 KB | normalisation statistics |
| `data/ensemble/*/ode_*.nc` | 4.9 MB | per-member ensemble intensity + embedded best track |
| `data/ensemble/*/FNV3_*_paired.csv` | 1.8 MB | Google FNV3 reference forecasts |
| `results/single_track/netcdf/*.nc` | ~1 MB | published per-storm output |
| `results/**/*.png`, `*.svg` | ~15 MB | published figures |

**On Google Drive** (links and SHA-256 sums in `data/manifest.json`):

| File | Size | Needed for |
|---|---|---|
| `fastml_training_data_2024_NA.tar` | 2.1 GB | running the CNN on the 2024 season |
| `chi_s_flossie.nc` | 17 MB | ventilation panel of the Flossie figure |
| `chi_s_priscilla.nc` | 35 MB | ventilation panel of the Priscilla figure |

The archive holds, per storm, `*_dataset.pkl` (track, observed intensity, FAST
scalars, ERA5 ventilation terms, environmental winds) and
`*_spatial_1000km.pkl` (the 72x72 storm-centred T/Q/U/V/Z on 7 levels plus SST
and MSLP that the CNN consumes). It extracts to `data/training_data/2024/` and
the tar is deleted afterwards.

`download_data.py` checks size **and** SHA-256 after each transfer, so a
truncated download fails loudly instead of quietly changing the results.

The raw GEFS archive (several TB) is **not** distributed and **not** required:
the per-member ventilation prediction and ODE integration are already stored in
the NetCDF files above.

## Ensemble case studies

Both cases drive FAST and FAST_ML with a GEFS-based synthetic track ensemble and
compare the intensity distribution against Google WeatherLab FNV3 and IBTrACS.

| Case | Initialisation | Configuration | Members |
|---|---|---|---|
| Flossie (EP062025) | 2025-06-29 12:00 | inherited forcing retained | 492 |
| Priscilla (EP162025) | 2025-10-04 18:00 | free run | 460 |

Priscilla is run without the inherited forcing, which removes the initial
momentum and leaves the ventilation response as the only driver of the spread.
The figures show the full ensemble, the ensemble mean, and the mean of the
strongest 10% of members; intensity ensembles are strongly right-skewed, so the
mean alone understates how well the distribution captures rapid intensification.

### How the ensemble products were made

The full chain, from raw forecast data to figure, is:

```
GEFS forecast fields  +  synthetic track ensemble  +  IBTrACS best track
          |
          |  (1) storm-following extraction: per-member ERA5-equivalent fields
          v
   per-member storm inputs
          |
          |  (2) the CNN of this repository, one forward pass per member
          v
   chi_s_<case>.nc        per-member chi and S, 480 steps        <- on Drive
          |
          |  (3) the FAST ODE of this repository, one integration per member,
          |      initialised from the best track; the ERA5 path is run
          |      alongside so both appear in the same file
          v
   ode_<case>.nc          per-member FAST and FAST_ML vmax,      <- in git
                          plus the embedded best track
          |
          |  (4) scripts/plot_ensemble_vs_google.py
          v
   the published figure
```

Steps 1-3 required the multi-terabyte GEFS archive and a GPU allocation, so
their outputs are distributed rather than their inputs. **Step 4 is what this
repository reproduces**, and it is the step the figures actually depend on:

```bash
python scripts/plot_ensemble_vs_google.py
```

This reads `ode_<case>.nc` for the intensity panel and `chi_s_<case>.nc` for the
ventilation panel, and reproduces both published figures exactly -- every peak
value in the legends and both ventilation means match.

Two details worth knowing if you compare against the figures in the manuscript:

- The intensity panel needs **only** the files tracked in git. Without the
  Drive-hosted `chi_s_*.nc` you still get a correct single-panel figure; pass
  `--no-vent-panel` to request that explicitly.
- `chi_s_priscilla.nc` archives 1000 members, but only the first 460 were
  successfully integrated and appear in `ode_priscilla_vp1p10_free.nc`. The
  plotting code truncates the ventilation archive to the members present in the
  ODE file, so both panels describe the same ensemble. Summarising all 1000
  would shift the ventilation means from 19.0/10.4 to 18.0/10.0.

## Citation

```bibtex
@article{xiao_fastml,
  title   = {Machine-learned ventilation for tropical cyclone intensity prediction},
  author  = {Xiao, Shijie},
  journal = {Journal of Advances in Modeling Earth Systems},
  note    = {Code: https://github.com/Shijie-Xiao/FAST_ML}
}
```

The FAST intensity model follows Emanuel (2012); the ventilation formulation
follows Tang and Emanuel (2012) and Lin et al. (2023). Google WeatherLab FNV3
forecasts are used under their terms of service; best-track data are from
IBTrACS.

## License

MIT, see `LICENSE`.
