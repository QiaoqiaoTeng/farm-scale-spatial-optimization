# Data dictionary and units

## `ML_Training_Data.csv` (2,100 rows)

| Column | Meaning | Unit / note |
|---|---|---|
| `Sim_ID` | Original scenario identifier | Integer |
| `Crop` | Crop | Maize, Soybean, Rice |
| `Soil_ID` | Soil category used upstream | Not supplied as a predictor to the uploaded optimization RF |
| `Density_plants_m2` | Planting density | plants/m² |
| `Fert_Weight_kg_ha` | 15-15-15 compound fertilizer mass | kg/ha, NOT elemental N |
| `Yield_kg_ha` | DSSAT-labeled yield result | kg/ha |
| `ET_mm` | DSSAT-labeled evapotranspiration | mm |
| `Status` | Source record simulation status | All uploaded rows `Success` |

## `Optimized_Farm_Plan_Continuous.csv` (54 rows)

`Field_ID` (1..54); `Area_ha` (as labeled in supplied source), `Optimized_Crop`, `Density_plants_m2`, `Fert_Weight_kg_ha`. The supplied code defines these same per-field areas. Confirm whether the `Area_ha` unit is physically correct with the study's source GIS records before public release.

## `Pareto_Front_Data.csv` (100 rows)

`Profit` (CNY), `WUE` (kg/m³), `Carbon` (kg CO2e) following the program's objective formulas. No decision-variable vectors accompany the 100 solutions; full per-field prescriptions for **each** Pareto solution therefore cannot be reconstructed from this file alone.
