"""Optional SHAP yield attribution (from provided RF, NOT a DSSAT rerun)."""
from pathlib import Path
import sys, argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from src.farmopt.core import load_models,FEATURES,CROPS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--samples',type=int,default=50)
    args=ap.parse_args()
    import shap
    df=pd.read_csv(ROOT/'data/source_outputs/ML_Training_Data.csv')
    models=load_models()
    out=ROOT/'results/reconstructed'
    out.mkdir(parents=True,exist_ok=True)
    for crop in CROPS:
        x=df[df.Crop==crop][FEATURES].sample(n=min(args.samples,700),random_state=42)
        sv=shap.TreeExplainer(models[crop]).shap_values(x)
        # For sklearn multi-output regression, modern SHAP has shape (samples, 2 features, 2 targets)
        a=np.asarray(sv)
        if a.ndim!=3 or a.shape[2]!=2:
            raise ValueError(f'Unexpected multioutput SHAP shape: {a.shape}')
        for k,target in enumerate(['Yield_kg_ha','ET_mm']):
            values=pd.DataFrame(a[:,:,k],columns=[f'SHAP_{z}' for z in FEATURES])
            values.insert(0,'source_index',x.index.to_numpy())
            values.to_csv(out/f'shap_{crop.lower()}_{target}.csv',index=False)
        print(f'{crop} SHAP computed for {len(x)} rows')
if __name__=='__main__':main()
