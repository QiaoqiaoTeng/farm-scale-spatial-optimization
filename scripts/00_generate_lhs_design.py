"""Regenerate a *new*, seed-controlled LHS scenario design (no DSSAT runs).

This is a reconstruction of the sampling logic found in the uploaded batch
script, NOT a recovery of the exact historical design from its archived CSV.
The original code used unseeded global np.random.choice for Soil_ID selection.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1]
BOUNDS={
    'Maize':((3.,18.),(900.,2700.)),
    'Soybean':((10.,35.),(250.,2000.)),
    'Rice':((15.,40.),(1800.,3400.)),
}
PADDY=set(range(10,22))|set(range(36,48))

def generate(n=700, seed=42):
    records=[]
    for ci,(crop,(density_bounds,fert_bounds)) in enumerate(BOUNDS.items()):
        sampler=qmc.LatinHypercube(d=2,seed=seed) # as coded originally, repeated by crop
        samples=qmc.scale(sampler.random(n=n),[density_bounds[0],fert_bounds[0]],
                          [density_bounds[1],fert_bounds[1]])
        pool=sorted(PADDY) if crop=='Rice' else sorted(set(range(1,55))-PADDY)
        # The original soil draws were not seeded. This controlled variant
        # uses independently seeded generators and may NOT match original IDs.
        soil_rng=np.random.default_rng(seed+ci)
        soil_ids=soil_rng.choice(pool,size=n,replace=True)
        for i,((density,fert),field) in enumerate(zip(samples,soil_ids),1):
            records.append(dict(Sim_ID=i+(ci*n),Crop=crop,
                Soil_ID=f'XY{int(field):08d}',Density_plants_m2=round(float(density),1),
                Fert_Weight_kg_ha=int(round(float(fert)))))
    return pd.DataFrame(records)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-per-crop',type=int,default=700)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',type=Path,default=ROOT/'results/recomputed_runs/new_lhs_design.csv')
    args=ap.parse_args()
    if args.n_per_crop<=0:raise ValueError('n-per-crop must be positive')
    out=generate(args.n_per_crop,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(f'New LHS design (not original): {len(out)} rows -> {args.output}')

if __name__=='__main__':main()
