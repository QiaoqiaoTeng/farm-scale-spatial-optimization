"""Reviewer model-loading smoke test, NOT an independent predictive validation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from src.farmopt.core import load_models,FEATURES,CROPS

def main():
    models=load_models()
    inputs=pd.read_csv(ROOT/'data/reviewer_test/inputs.csv')
    expected=pd.read_csv(ROOT/'data/reviewer_test/expected_model_predictions.csv')
    passes=0
    for crop in CROPS:
        inp=inputs[inputs.Crop==crop].sort_values('Sim_ID')
        exp=expected[expected.Crop==crop].sort_values('Sim_ID')
        pred=models[crop].predict(inp[FEATURES])
        target=exp[['Expected_Predicted_Yield_kg_ha','Expected_Predicted_ET_mm']].to_numpy()
        good=np.allclose(pred,target,rtol=1e-8,atol=1e-7)
        print(f'{crop}: {pred.shape[0]} examples, 2 outputs, {"PASS" if good else "FAIL"}')
        passes+=int(good)
    if passes!=3:raise SystemExit('FAIL: model outputs differ from expected.')
    print('All three model-loading checks passed.')
if __name__=='__main__':main()
