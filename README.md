# FAST-ML: A Hybrid Physics–Machine Learning Framework for Tropical Cyclone Intensity Forecasting

Shijie Xiao<sup>1,4</sup>, Jonathan Lin<sup>2</sup>, Thomas Ehrmann<sup>3</sup>, Ali Sarhadi<sup>1,4</sup>

<sup>1</sup> School of Earth and Atmospheric Sciences, Georgia Institute of Technology, Atlanta, GA, USA
<sup>2</sup> Department of Earth and Atmospheric Sciences, Cornell University, Ithaca, NY, USA
<sup>3</sup> Sandia National Laboratories, Albuquerque, NM, USA
<sup>4</sup> School of Interactive Computing, Georgia Institute of Technology, Atlanta, GA, USA

Code and data accompanying the manuscript submitted to *Journal of Advances in
Modeling Earth Systems* (JAMES). This is an **inference-only** release: it
reproduces the published results but does not contain the training loop or the
code that derives the ERA5 ventilation diagnostics from raw reanalysis.

What the paper contributes, and what this repository lets you check:

- A hybrid framework in which a CNN diagnoses only the ventilation index and an
  ODE does all the temporal integration, so the physics constrains the forecast
  and the learned component stays interpretable.
- A mean 4.6 kt RMSE reduction over pure FAST across the 2024 North Atlantic
  test season, improving 10 of 12 storms, with no change to the dynamics.
- Ensemble forecasts that beat Google WeatherLab FNV3 on recent 2024-2025
  rapidly intensifying eastern Pacific and North Atlantic hurricanes, where
  purely data-driven models damp the intensity tail.

## What the model does

FAST predicts tropical cyclone intensity by integrating a coupled ODE for the
axisymmetric wind `V` and a moisture variable `m`. Its dominant source of
uncertainty is the **ventilation index** `chi * S`, the rate at which low
entropy environmental air is stirred into the core, which is conventionally
diagnosed from reanalysis. FAST-ML replaces that diagnosis with a two-stream
CNN applied to storm-centred ERA5 fields and changes nothing else: both paths
share one ODE, one track, one set of scalars, one drag field and one
initialisation procedure, so any difference in skill is attributable to the
ventilation diagnosis alone.

![FAST-ML framework](img/Figure_2.png)

*Figure 2. (A) Storm-centred ERA5 fields and FAST scalars feed a two-stream CNN
that diagnoses wind shear `S` and ventilation `chi`; these enter the FAST
physics layer, which integrates the state forward. (B) At forecast time a track
model and forecast fields drive the same network to produce an intensity
ensemble.*

## Reproduction

Three levels, each self-contained. Pick the one you need.

```bash
git clone https://github.com/Shijie-Xiao/FAST_ML.git
cd FAST_ML
pip install -r requirements.txt
```

**Level 1 — figures from the archived output. No download, no GPU, ~1 min.**

```bash
python scripts/run_single_track.py --replot-only
python scripts/plot_ensemble_vs_google.py --no-vent-panel
```

**Level 2 — the published ensemble figures in full. 53 MB, ~2 min.**

```bash
python scripts/download_data.py --ensemble
python scripts/plot_ensemble_vs_google.py
```

**Level 3 — run the model end to end. 2.1 GB, ~5 min on a GPU.**

```bash
python scripts/download_data.py --single-track
python scripts/run_single_track.py --years 2024 --basin NA
```

Level 3 reruns the CNN and the ODE over the 2024 North Atlantic test season and
rewrites the NetCDF, figures and metrics from scratch:

```
Storms evaluated          : 12  (7 skipped)
Mean RMSE, FAST           : 15.42 kt
Mean RMSE, FAST_ML        : 10.78 kt
Mean improvement          : +4.64 kt
Storms where FAST_ML wins : 10/12
```

Seven of the nineteen 2024 storms are excluded by the evaluation criteria and
the run reports why for each: they either never reach the 45 kt forecast
threshold or have under 72 h of forecast after it. To see which inputs are
present without downloading anything, run `python scripts/download_data.py
--verify`.

For one storm:

```bash
python scripts/run_single_track.py --years 2024 --storm MILTON
```

## Results

![Per-storm evaluation](img/Figure_3.png)

*Figure 3. (A) Per-year mean RMSE across 2003-2024 with the validation and test
years marked. (B, C) Per-storm RMSE for the 2022 and 2023 validation seasons.
(D) Per-storm RMSE for the 2024 test season. Green is FAST, blue is FAST-ML;
labels give the RMSE reduction in knots.*

Panel D is the season reproduced by this repository. RMSE is scored against the
IBTrACS best track over the full forecast period, in knots:

| Storm | Observed peak | FAST | FAST-ML | Improvement |
|---|---:|---:|---:|---:|
| BERYL | 145 | 12.78 | 10.35 | +2.42 |
| DEBBY | 70 | 18.60 | 16.32 | +2.28 |
| ERNESTO | 85 | 10.31 | 10.68 | −0.37 |
| FRANCINE | 90 | 4.96 | 2.92 | +2.04 |
| HELENE | 120 | 9.02 | 6.85 | +2.17 |
| ISAAC | 90 | 28.97 | 9.17 | +19.80 |
| KIRK | 130 | 20.59 | 14.90 | +5.69 |
| LESLIE | 90 | 21.78 | 14.11 | +7.67 |
| MILTON | 155 | 29.24 | 19.81 | +9.43 |
| OSCAR | 75 | 8.40 | 5.70 | +2.70 |
| PATTY | 55 | 3.21 | 6.03 | −2.81 |
| RAFAEL | 105 | 17.20 | 12.52 | +4.68 |
| **mean** | | **15.42** | **10.78** | **+4.64** |

Bias, correlation and peak intensity per storm are in
`results/single_track/storm_metrics.csv`, and the full time series for each
storm is in `results/single_track/netcdf/`.

## Ensemble forecasting

The same network runs in ensemble mode: a GEFS-based synthetic track ensemble
drives FAST and FAST-ML member by member, giving a full intensity distribution
that we compare against the Google WeatherLab FNV3 ensemble and IBTrACS.

**This is where the hybrid framework is most useful.** On recent (2024-2025)
rapidly intensifying eastern Pacific and North Atlantic hurricanes, the FAST-ML
ensemble captures observed peak intensity that FNV3 under-forecasts. A purely
data-driven model is trained towards the conditional mean and systematically
damps the tail; retaining the ODE means the ensemble spread is generated by
physics, and the ML component only has to diagnose ventilation. The two cases
released here are representative:

| Case | Observed peak | FAST-ML mean | FAST-ML top 10% | FNV3 mean | FNV3 top 10% |
|---|---:|---:|---:|---:|---:|
| Flossie (EP06 2025) | 105 | **97** | **113** | 79 | 101 |
| Priscilla (EP16 2025) | 100 | **95** | **108** | 85 | 98 |

Peak intensity in knots; "mean" is the peak of the ensemble-mean track and
"top 10%" that of the strongest decile of members. In both cases the FAST-ML
ensemble mean is within 5-8 kt of the observed peak against 15-26 kt for the
FNV3 mean, and the FAST-ML top decile brackets the observed peak while the FNV3
top decile still falls short of it. Intensity ensembles are strongly
right-skewed, so the top decile is reported alongside the mean: it is the part
of the distribution that matters for a rapid-intensification warning.

| Case | Initialisation | Configuration | Members | FAST mean / top 10% |
|---|---|---|---|---|
| Flossie (EP062025) | 2025-06-29 12:00 | inherited forcing retained | 492 | 118 / 129 |
| Priscilla (EP162025) | 2025-10-04 18:00 | free run | 460 | 81 / 96 |

The pure-physics FAST baseline misses in both directions — over-forecasting
Flossie by 13 kt and under-forecasting Priscilla by 19 kt — so the improvement
is not a bias correction in one direction but a better diagnosis of ventilation.
Priscilla is run without the inherited forcing, which removes the initial
momentum and leaves the ventilation response as the only driver of the spread.

![Flossie ensemble](results/ensemble/flossie_midfinit_finit_vs_google.png)

*Flossie (EP06 2025), initialised 2025-06-29 12:00 with inherited forcing
retained. Left: the 492-member FAST-ML intensity ensemble with its mean and top
decile, against the 50-member Google FNV3 ensemble and the IBTrACS best track.
Right: the corresponding per-member ventilation index.*

![Priscilla ensemble](results/ensemble/priscilla_free_vs_google.png)

*Priscilla (EP16 2025), initialised 2025-10-04 18:00 as a free run. Same layout.
With no inherited forcing the entire spread comes from the ventilation response.*

Provenance of these products, since the raw GEFS archive is too large to
distribute:

```
GEFS fields + synthetic track ensemble + IBTrACS best track
   |  (1) storm-following extraction, per member
   |  (2) the CNN of this repository, one forward pass per member
   v      -> chi_s_<case>.nc     per-member chi and S      [hosted, 53 MB]
   |  (3) the FAST ODE of this repository, one integration per member
   v      -> ode_<case>.nc       per-member FAST and FAST-ML vmax  [in git]
   |  (4) scripts/plot_ensemble_vs_google.py
   v      -> the published figure
```

Steps 1-3 needed the multi-terabyte GEFS archive and a GPU allocation, so their
outputs are distributed rather than their inputs. Step 4 is what this repository
reproduces, and it is the step the figures depend on. The intensity panel needs
only files tracked in git; `chi_s_*.nc` supplies the ventilation panel. Note
that `chi_s_priscilla.nc` archives 1000 members but only the first 460 were
integrated and appear in the ODE file, so the plotting code truncates the
ventilation archive to the members actually plotted.

## Data

Everything needed to redraw the published figures is in git (about 30 MB).
Only running the CNN, or drawing the ensemble ventilation panels, needs the
externally hosted files.

**In git:** model weights (`ckpt/`, 4.7 MB), drag coefficient field
(`data/Cd.nc`, 8.3 MB), normalisation statistics (<1 KB), ensemble ODE output
and Google FNV3 reference (`data/ensemble/`, 6.7 MB), and the published NetCDF
and figures (`results/`, ~16 MB).

**Hosted externally**, with links and SHA-256 sums in `data/manifest.json`:

| File | Size | Needed for |
|---|---|---|
| `fastml_training_data_2024_NA.tar` | 2.1 GB | running the CNN on the 2024 season |
| `chi_s_flossie.nc` | 17 MB | ventilation panel, Flossie |
| `chi_s_priscilla.nc` | 35 MB | ventilation panel, Priscilla |

The archive holds, per storm, `*_dataset.pkl` (track, observed intensity, FAST
scalars, ERA5 ventilation terms, environmental winds) and
`*_spatial_1000km.pkl` (the 72×72 storm-centred T/Q/U/V/Z on 7 levels plus SST
and MSLP that the CNN consumes). `download_data.py` checks size and SHA-256
after each transfer, so a truncated download fails loudly rather than quietly
changing the results.

Normalisation statistics are the one thing that cannot be derived from what is
distributed here, because they come from the training set. They are committed
directly as `data/spatial_stats_train2003_2022.pkl`, and
`scripts/compute_spatial_stats.py` documents and regenerates them if you have
the training archive.

## Reproduction fidelity

Verified by cloning this repository into an empty directory, fetching the hosted
files and rerunning everything. The NetCDF output is bit-identical to the
committed results for every storm and variable, and `storm_metrics.csv` and the
ensemble legend values match exactly.

Figures are byte-identical when regenerated on the same machine. Across
machines, matplotlib's anti-aliasing is not bit-reproducible: a fresh clone
reproduced 3 of 8 figures byte-for-byte and the rest differed by one greyscale
level on tens of pixels out of 2.2 million. Compare the NetCDF, not the PNGs.

Two deliberate choices make this code differ in the last digits from the
internal version that produced the manuscript. Track coordinates are kept in
float64 throughout, where the internal pipeline briefly cast them to float32,
which perturbs the Coriolis parameter and drag lookup at the 1e-4 kt level; this
version is the more accurate of the two. And the `vp_kts` variable in the NetCDF
is the median-filtered series the ODE actually integrates, where the internal
version reported the unfiltered series alongside a filtered integration.

## Method summary

**Time axis.** The forecast starts when the best track first reaches 45 kt. The
preceding 48 h initialise the ODE, so every sequence is cut to begin 48 h before
that threshold.

**Initialisation.** Over the 48 h window, `V` is nudged onto the observation and
the residual `F = observed acceleration − physics RHS` is recorded, while `m`
spins up freely. From `t_start` the integration is free, with the inherited
forcing decaying as `F * exp(−2 * (lead / 24 h)^2)`. Without this the forecast
would discard the momentum the storm already had at initialisation.

**Observation inversion.** The best track reports the maximum sustained wind,
which contains the storm translation and the shear-induced asymmetry, whereas
the ODE state is the axisymmetric mean. Observations are inverted to
axisymmetric form before being used as a target, and model output is converted
back for scoring.

**Ventilation calibration.** The relevant quantity is the driest air reaching
the core, not the annulus mean. Both paths apply the same
mean-to-90th-percentile factor and the same Atlantic ceiling, so the ERA5 and
CNN ventilation indices live on one common scale.

**The two streams.** `S` is predicted from the wind and geopotential fields
(U, V, Z) and `chi` from the thermodynamic fields (T, Q, SST, MSLP). The split
follows the physics: `S` is a shear quantity and `chi` an entropy deficit.
Neither stream is recurrent — both are per-timestep diagnostics — which keeps
the network from learning the observed intensity trajectory and forces all
temporal evolution to come from the ODE.

## Repository layout

```
fastml/
  config.py        Constants and thresholds, shared by both paths
  model.py         Two-stream CNN that diagnoses chi and S  (inference only)
  fast_physics.py  The FAST ODE, 48 h initialisation, wind conversion (NumPy)
  data.py          Storm loading, time alignment, normalisation
  inference.py     Runs FAST and FAST-ML over one storm
  netcdf_io.py     The single-track NetCDF product
  plotting.py      Single-track and ensemble figures
scripts/
  run_single_track.py         Single-track hindcast driver
  plot_ensemble_vs_google.py  Ensemble case-study figures
  compute_spatial_stats.py    Regenerates the normalisation statistics
  download_data.py            Fetches the externally hosted files
ckpt/   Released weights (4.7 MB, 237 tensors, strict load)
data/   Drag field, normalisation statistics, ensemble inputs, manifest
img/    Manuscript figures
results/ Published NetCDF output and figures
```

## Data sources

IBTrACS v04 best-track data (NOAA NCEI), ERA5 reanalysis (Copernicus Climate
Change Service), GEFS reforecasts (NOAA), and Google WeatherLab FNV3 ensemble
forecasts.

## Citation

```bibtex
@article{xiao_fastml,
  title   = {{FAST-ML}: A Hybrid Physics--Machine Learning Framework for
             Tropical Cyclone Intensity Forecasting},
  author  = {Xiao, Shijie and Lin, Jonathan and Ehrmann, Thomas and
             Sarhadi, Ali},
  journal = {Journal of Advances in Modeling Earth Systems},
  note    = {Submitted},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
