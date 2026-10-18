"""Refreeze the three two-output RF estimators; Python 3.13 + pinned versions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from src.farmopt.core import FEATURES,TARGETS,CROPS

def main():
    df=pd.read_csv(ROOT/'data/source_outputs/ML_Training_Data.csv')
    (ROOT/'models').mkdir(exist_ok=True)
    for crop in CROPS:
        d=df.loc[df.Crop==crop].dropna()
        model=RandomForestRegressor(n_estimators=100,random_state=42)
        model.fit(d[FEATURES],d[TARGETS])
        dest=ROOT/'models'/f'{crop.lower()}_yield_et_rf.joblib'
        joblib.dump(model,dest,compress=3)
        print(f'{crop}: {len(d)} rows -> {dest}')
if __name__=='__main__': main()
