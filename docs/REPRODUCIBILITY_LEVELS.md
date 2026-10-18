# What this release reproduces

| Level | Status | Command |
|---|---|---|
| Direct load of trained, frozen RF models | Verified in package build | `python scripts/02_test_models.py` |
| Rebuild the 3 two-output RFs using the supplied 2,100-row CSV | Available, uses the optimizer's fit specification | `python scripts/01_train_models.py` |
| New 80/20 holdout metrics | Calculated from uploaded CSV, **not proven equal** to published validation | `results/validation/new_holdout_metrics.csv` |
| Reconstruct objectives for the archived single 54-field plan | Available, **derived** | `python scripts/03_reconstruct_plan_metrics.py` |
| Plot archived 100-row Pareto and 54-field prescription | Verified | `python scripts/04_plot_archived_results.py` |
| Compute optional SHAP for 3 saved multi-output RFs | Available; requires `requirements-optional.txt` | `python scripts/05_shap_optional.py` |
| Rerun 138-variable 3-objective NSGA-II | **Optional, not executed during build**; requires pymoo, can be computationally expensive | `python scripts/06_optimize_optional.py --quick` |
| Full upstream WGEN→DSSAT→GLUE→LHS regeneration | **Unavailable** (source input/code absent) | — |
| Official manuscript figures and complete chosen decision strategies | **Not fully reproduced** | — |
