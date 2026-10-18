# Farm-scale spatial optimization: review reproducibility package

**Associated manuscript:** *An Integrated Framework Coupling Mechanistic and Interpretable Surrogate Models for Farm-Scale Multi-Objective Spatial Optimization*.

This is a **source-grounded downstream reproducibility package** created from the uploaded optimization/visualization scripts, archived outputs, and supplementary DSSAT experiment-preparation files. It is **not yet a complete end-to-end WGEN–DSSAT–GLUE–LHS archive**; however, the author-supplied LHS/DSSAT batch script and FileX templates are now included as audited experimental preparation assets. See [`docs/METHODOLOGY_GAPS.md`](docs/METHODOLOGY_GAPS.md) before journal submission.

## Data and model inventory

- `data/source_outputs/ML_Training_Data.csv`: 2,100 source scenarios, 700 each for Maize/Soybean/Rice.
- `data/source_outputs/Optimized_Farm_Plan_Continuous.csv`: archived plan for 54 fields.
- `data/source_outputs/Pareto_Front_Data.csv`: 100 archived objective vectors.
- `models/`: **3 crop-specific, two-output** Random Forest regressors that predict both `Yield_kg_ha` and `ET_mm`. They reproduce the provided optimizer's estimator architecture, not necessarily the separate models and CV procedure described in the manuscript.
- `data/reviewer_test/`: 15 stable input samples and expected predictions. These are *model loading tests*, not external validation.
- `results/validation/new_holdout_metrics.csv`: new 80/20 test computed from the uploaded dataset; do not quote these as manuscript results without checking the original analysis.

## Quick start (verified review path)

Use **Python 3.13** and create a dedicated virtual environment. The serialized model files were generated under scikit-learn **1.8.0**. Scikit-learn joblib models cannot be assumed to load across arbitrary package versions.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/02_test_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Unix/macOS:

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/02_test_models.py
./.venv/bin/python -m unittest discover -s tests -v
```

Expected test summary: **Maize PASS, Soybean PASS, Rice PASS** (two outputs per model).

## DSSAT input preparation (new; offline verified only)

The following new files provide auditability of the **pre-DSSAT experimental setup**, without claiming end-to-end DSSAT reproduction:

- `dssat/source_experiments/`: author-supplied `.MZX` / `.SBX` / `.RIX` input examples.
- `dssat/templates/`: author-supplied placeholderized crop input templates.
- `src/legacy/additional_sources/`: all eight newly uploaded Python variants; workstation-specific paths in the legacy DSSAT batch source were sanitized.
- `scripts/00_prepare_dssat_examples.py`: offline token/fixed-width verified preparation of **derived** sample DSSAT input files.
- `scripts/00_generate_lhs_design.py`: **new** deterministic LHS plan reconstructed from the source sampler. It does not recover the historical soil draws.

```bash
python scripts/00_prepare_dssat_examples.py
python scripts/00_generate_lhs_design.py --n-per-crop 700 --seed 42
```

For methodological and redistribution limitations read [`dssat/README.md`](dssat/README.md) and [`docs/METHODOLOGY_GAPS.md`](docs/METHODOLOGY_GAPS.md).

## Other review workflows

```bash
python scripts/01_train_models.py             # Refit/overwrite 3 RF models (same input/targets as optimizer)
python scripts/03_reconstruct_plan_metrics.py # Recompute derived plan metrics; clearly labeled RECONSTRUCTED
python scripts/04_plot_archived_results.py    # Generate 2 figures strictly from supplied CSVs
```

Optional analyses (not part of the minimal verified smoke test):

```bash
pip install -r requirements-optional.txt
python scripts/05_shap_optional.py --samples 50
python scripts/06_optimize_optional.py --quick
# full NSGA-II is much slower:
python scripts/06_optimize_optional.py
```

A *new* optimization run is not guaranteed to produce the identical 100 archived Pareto rows, and writes only into `results/recomputed_runs/`. For the exact original scripts, see `src/legacy/` (the visualization script expects an unavailable metrics CSV).

## Repository structure

```text
configs/parameters.json            Source-extracted economic/agronomic/optimization constants
src/farmopt/core.py              Inference, 138-variable decoding and objective reconstruction
src/legacy/                      Main original scripts and 8 additional historical variants (batch paths sanitized)
scripts/                         Tested and optional workflow entry points
data/source_outputs/             Supplied CSVs and source-extracted field metadata
data/reviewer_test/              Model-loading input/output checks
models/                          3 frozen dual-output RF estimators + metadata
results/                         New holdout diagnostics, plots, and derived results
tests/                           Automated integrity/model tests
docs/                            Audit, data dictionary, provenance manifest, reproducibility tiers
dssat/                           Original experimental FileX and templates + offline preparation tests
```

## Scope and scientific caveats

- The source optimizer uses `Density_plants_m2` and `Fert_Weight_kg_ha` only; `Soil_ID` is not part of its RF predictors.
- Its median-filter mutation operates on sequential **field identifiers**, not the confirmed spatial adjacency graph of real plots. Synthetic plot panels in the legacy visualization must not be passed off as true measured spatial evidence.
- The original **LHS and DSSAT batch-preparation code and three crop FileX templates are supplied**, and a safe offline template test is included. Independent DSSAT execution remains unverified: necessary soil/weather/cultivar inputs and the executable are absent. WGEN/GLUE and ARIMA workflows remain unavailable.
- The archived Pareto objectives do not include the per-solution 138-variable decision vectors.
- The field-area values/units and the published metrics must be verified by the research authors.

## Citation and release

`CITATION.cff` records manuscript authors, and this repository can be frozen as `v0.1-review` following scientific audit. Final code and data licensing requires authorization from the coauthors or institution; see `LICENSE_DECISION.md`. DO NOT include private manuscripts, peer review correspondence, proprietary executables, or unauthorised upstream datasets in a public repository.
