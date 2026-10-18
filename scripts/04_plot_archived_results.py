"""Draw only plots from the actual archived Pareto/plan CSV files."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
OUT.mkdir(parents=True,exist_ok=True)
pareto=pd.read_csv(ROOT/'data/source_outputs/Pareto_Front_Data.csv')
plan=pd.read_csv(ROOT/'data/source_outputs/Optimized_Farm_Plan_Continuous.csv')
fig,ax=plt.subplots(figsize=(7.2,5.2))
pts=ax.scatter(pareto.Profit/1e8,pareto.Carbon/1e8,c=pareto.WUE,s=25)
ax.set(xlabel='Total Profit (1e8 CNY)',ylabel='Total Carbon (1e8 kg CO2e)',title='Archived Pareto solutions (100 rows)')
fig.colorbar(pts,ax=ax,label='WUE (kg/m³)')
fig.tight_layout();fig.savefig(OUT/'archived_pareto_projection.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
for crop,grp in plan.groupby('Optimized_Crop'):
    ax.scatter(grp.Field_ID,grp.Density_plants_m2,label=crop,s=30)
ax.legend();ax.set(xlabel='Field ID (not geographic coordinate)',ylabel='Density (plants/m²)',title='Archived optimized field prescription')
fig.tight_layout();fig.savefig(OUT/'archived_field_density.png',dpi=180);plt.close(fig)
print('Saved 2 archive-grounded figures to', OUT)
