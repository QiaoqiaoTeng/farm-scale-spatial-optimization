"""Optional full 138-dimensional optimization; requires pymoo 0.6.x.

The optimized runs generated here are NEW runs, not claimed identical to
provided 100-row Pareto CSV. They will not overwrite archived source data.
"""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from src.farmopt.core import decode_solution,evaluate_plan,load_config,load_models,check_constraints


def main():
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.mutation import Mutation
    from pymoo.operators.mutation.pm import PM
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true',help='3 generations, 12 individuals: pipeline check only')
    parser.add_argument('--output',default='results/recomputed_runs')
    args=parser.parse_args()
    config=load_config()
    models=load_models()
    opt=config['optimizer']
    gens=3 if args.quick else opt['generations']
    pop=12 if args.quick else opt['population']

    class FarmProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=138,n_obj=3,n_ieq_constr=2,xl=np.zeros(138),xu=np.ones(138))
        def _evaluate(self,x,out,*_,**__):
            plan=decode_solution(x,config)
            obj=evaluate_plan(plan,models,config)[1]
            c=check_constraints(plan,config)
            out['F']=[-obj['Profit'],-obj['WUE'],obj['Carbon']]
            out['G']=[config['constraints']['minimum_maize_area_ha_as_coded']-c['maize_area_ha'],
                      config['constraints']['minimum_soybean_area_ha_as_coded']-c['soybean_area_ha']]

    class MemeticSpatialMutation(Mutation):
        def __init__(self):
            super().__init__()
            self.pm=PM(prob=1/138,eta=20)
        def _do(self,problem,X,**kwargs):
            a=self.pm._do(problem,X,**kwargs)
            for i in range(a.shape[0]):
                if np.random.rand()<0.8:
                    a[i,30:84]=np.clip(median_filter(a[i,30:84],size=3),0,1)
                    a[i,84:138]=np.clip(median_filter(a[i,84:138],size=3),0,1)
            return a

    # Both explicit optimizer and legacy original are seeded with 1.
    np.random.seed(int(opt['seed']))
    result=minimize(FarmProblem(),NSGA2(pop_size=pop,mutation=MemeticSpatialMutation(),eliminate_duplicates=True),
                    ('n_gen',gens),seed=int(opt['seed']),verbose=False)
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    if result.F is None:
        raise RuntimeError('No feasible Pareto front produced. This can occur in quick mode. The archived front remains intact.')
    front=pd.DataFrame({'Profit':-result.F[:,0],'WUE':-result.F[:,1],'Carbon':result.F[:,2]})
    front.to_csv(output/('quick_pareto_new_run.csv' if args.quick else 'pareto_new_run.csv'),index=False)
    print('New run complete,',len(front),'solutions. Archived Pareto unchanged.')
if __name__=='__main__':main()
