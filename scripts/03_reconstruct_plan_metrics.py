"""Derived reconstruction, not a substitute for missing original metrics output."""
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.farmopt.core import evaluate_plan,check_constraints

def main():
    p=pd.read_csv(ROOT/'data/source_outputs/Optimized_Farm_Plan_Continuous.csv')
    values,score=evaluate_plan(p)
    dest=ROOT/'results/reconstructed'
    dest.mkdir(parents=True,exist_ok=True)
    values.to_csv(dest/'plan_metrics_reconstructed_from_rf.csv',index=False)
    summary={'source':'RECONSTRUCTED FROM PROVIDED PLAN; not an original archived run', 'objectives':score,'constraints':check_constraints(p)}
    (dest/'reconstructed_objectives.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
