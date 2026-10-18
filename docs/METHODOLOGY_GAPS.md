# Verification notes and scientific limitations

Read before sharing this package with a journal. These points are grounded in the uploaded scripts, CSVs and the cleaned manuscript.

## Confirmed

1. The uploaded training table contains **2,100** successful records: 700 per crop. The uploaded optimized plan has **54** rows, and the archived Pareto table has **100** rows.
2. The production optimization script (`src/legacy/original_farm_optimization.py`) fits **three RandomForestRegressor objects, each with two outputs (Yield, ET)**, on all available rows (`n_estimators=100`, `random_state=42`). The cleaned manuscript states *two independent sets* per crop and 8:2 split + GridSearchCV with 10-fold CV. **These are not identical model-selection/training procedures.** The review-package models reproduce the uploaded optimizer implementation, not the unprovided separate models.
3. `Soil_ID` is present in the training table, but the uploaded optimizer RF uses only two predictors: planting density and compound fertilizer mass. Therefore, soil-specific yield responses are not inferred from `Soil_ID` by these saved RF models. Treat any claim of soil-aware surrogate prediction as unverified pending additional code.
4. The uploaded spatial smoothing mutation calls SciPy `median_filter(..., size=3)` on a one-dimensional sequence of **field IDs**, not a genuine GIS adjacency matrix or spatial polygon neighborhood. This supports **index-neighborhood smoothing**, not a proven geographically contiguous management pattern.
5. The visualization script builds a `QuickFarmOpt` toy problem (10 continuous variables, 2 objectives, simple algebraic profit/carbon) to draw its convergence panel. That graphic must **not** be described as convergence diagnostics of the real 138-variable 3-objective farm optimizer. Additional smoothing illustrations use a synthetic 6×9 grid, not observed geographic field adjacency.
6. The original visualization script requires `All_Metrics_Data_Continuous.csv`, which was not among the three supplied CSVs. The replacement `scripts/04_plot_archived_results.py` draws only figures supported by archived CSVs. The reconstructed metrics script separately labels computed numbers as a reconstruction rather than original exported values.
7. The original optimizer has 138 variables (30 dryland crop choices, 54 density values, 54 fertilizer values), 100 population, 200 generations, objectives (profit maximize, WUE maximize, carbon minimize), and minimum crop-area constraints. The supplied 54 field areas total **47,609 in the CSV's 'Area_ha' units**; this large total and whether it matches actual land area requires external confirmation.
8. The 100 archived Pareto rows contain only objective triples and cannot recreate all 100 associated decision vectors.
9. **New supporting materials now available:** an author-supplied LHS design and DSSAT batch-execution helper, fixed-width FileX source examples, and placeholder templates for the three crops. Their template-rendering step is offline tested. However, the **WGEN**, **GLUE**, **ARIMA**, the genuine soil/weather/cultivar inputs and the complete DSSAT installation are still not available, so upstream simulation results were **not** independently regenerated. The original LHS soil selections used unseeded random draws and the batch runner includes environment-specific paths. See `dssat/README.md`.
10. The provided scripts' seeds and the newly saved models permit limited downstream reproducibility; full equivalence with manuscript figures/tables and previously run raw DSSAT outputs requires author verification. The test accuracy reported by this package is **a new 80/20 split calculation**, NOT automatically the same training/test split and hyperparameter search reported in the manuscript.

## Editorial action needed

- Ask the authors to identify the exact manuscript model-selection code, 10-fold CV results, and trained independent RF estimators if those claims are retained.
- Supply the true 54-field geography/adjacency and explain or revise the spatial-smoothing claims.
- The FileX source and placeholder files are now deposited as draft material. Obtain permission to redistribute and additionally archive the **real** cultivar/soil/weather files, GLUE calibration and original scenario-ID mapping if full pipeline reproducibility is required.
- Reconcile the meaning of `Area_ha`, the role of `Soil_ID`, and all figure legends/units.
- Confirm output reference values and create a frozen `v1.0-review` Release only after the above review.

## Additional notes from the newly uploaded scripts

- `rf-新增可视化(1).py` depends on `N_Fertilizer_kg_ha` and `N_Leach_kg_ha`; those columns are not in the supplied `ML_Training_Data.csv`. It is archived as a **historical variant**, not a tested current workflow.
- `Part3_Algorithm_Diagnostics(1).py` uses a synthetic `QuickFarmOpt` to generate figure traces. Its results **must not** be described as actual convergence of the 54-field optimizer.
- `可视化p2-p3(1).py` has an automatic mock-data fallback when training CSV is missing. The mock path should be excluded from any figure represented as actual experimental results.
- `地块可视化(1).py` includes a 54-field area bar chart but **not geographic polygon geometries**; it does not resolve true adjacency.
- The original LHS generator calls the scipy LatinHypercube sampler with fixed `seed=42` but uses an **unseeded** NumPy global RNG to choose Soil_ID. The provided new design generator makes an explicit reproducibility choice that does not recover the original historical soil IDs.
- The DSSAT batch source parses `HWAM` and `ETCM`; their scientific units must be independently confirmed before promising `Yield_kg_ha` and `ET_mm`.
