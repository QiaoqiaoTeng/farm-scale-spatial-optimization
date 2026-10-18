# DSSAT v4.8 support: experiment FileX and LHS batch preparation

This directory contains the supplied **three original crop experiment FileX
examples** (`source_experiments/`) and their **placeholderized template
variants** (`templates/`). The files were supplied by the authors and have
not been validated against a complete DSSAT installation in this review package.

| Crop | Source experiment | Placeholder template | Model code in batch script |
|---|---|---|---|
| Maize | `CHXF2401.MZX` | `TPL_MAIZE.MZX` | `MZCER048` |
| Soybean | `CHXF2402.SBX` | `TPL_SOY.SBX` | `SBGRO048` |
| Rice | `CHXF2403.RIX` | `TPL_RICE.RIX` | `RICER048` |

## What the newly supplied `批量生成文件(1).py` actually implements

- Latin Hypercube designs with two sampled agronomic variables per crop, 700
  rows per crop; `scipy.stats.qmc.LatinHypercube(d=2, seed=42)`.
- Selection of a soil/field ID from a crop-specific pool. **The original code
  uses global `np.random.choice` without a seed**. It therefore cannot
  reproduce the original field-ID sequence solely from its configuration.
- Preparation of fixed-width crop experiment files by replacing soil, density,
  and 15-15-15 compound-fertilizer placeholders.
- Calls to an external, separately installed DSSAT v4.8 executable and parses
  selected columns of `Summary.OUT`.

## Safe offline reproducibility checks

From repository root:

```bash
python scripts/00_prepare_dssat_examples.py
python scripts/00_generate_lhs_design.py --n-per-crop 700 --seed 42
python -m unittest discover -s tests -v
```

`00_prepare_dssat_examples.py` writes three **derived** FileX examples in
`results/generated_dssat_examples/` from real archived input rows. It
**does not call DSSAT**. It validates token uniqueness, fixed width and CRLF
line endings. It also does not assert that source FileX model settings are
scientifically calibrated.

`00_generate_lhs_design.py` saves a **new deterministic LHS design** in
`results/recomputed_runs/`. This is a reproducibility helper, **not** the
historical raw LHS table that produced the deposited 2,100 DSSAT records.

## Dependencies still required for real crop-model runs

The uploaded FileX contain references to plant/cultivar, soil, and weather
records that are external to the deposited templates. **The soil profiles,
weather series (including WGEN realization), calibrated cultivar files,
DSSAT binary, run configuration and execution environment are not bundled.**
The GLUE calibration workflow and its parameter posteriors are also absent.
Do not claim that the entire WGEN–GLUE–DSSAT simulation can be reproduced from
this repository alone.

Author verification required before journal use:

1. DSSAT version, cultivar/soil/weather identifiers and their licensed access.
2. Verify `Summary.OUT` HWAM/ETCM column definitions and **units** against
   the installed model; the original batch script assumes ET is already `mm`.
3. Confirm fixed-width plant population and fertilizer field units, and whether
   P/K conversions should use elemental or oxide basis for the chosen DSSAT
   configuration.
4. Reconcile the archived training table's scenarios, soil ID random assignment,
   and original DSSAT run results with the supplied helper source.
5. Confirm redistribution rights for the provided FileX/parameter content.

### Provenance of historical batch source

`src/legacy/additional_sources/lhs_dssat_batch_original_path_sanitized.py`
retains the algorithmic content of the uploaded batch script, but workstation-
specific absolute paths were replaced by neutral placeholders prior to
packaging. It is a **legacy reference**, not the supported test entrypoint.
