"""Reconstruction of the objective function in original_farm_optimization.py.

IMPORTANT: plot-number median_filter is NOT real geospatial adjacency.
This module does not implement WGEN, DSSAT, GLUE, or LHS.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[2]
FEATURES = ['Density_plants_m2', 'Fert_Weight_kg_ha']
TARGETS = ['Yield_kg_ha', 'ET_mm']
CROPS = ('Maize', 'Soybean', 'Rice')


def load_config():
    return json.loads((BASE / 'configs' / 'parameters.json').read_text(encoding='utf-8'))


def load_models():
    return {crop: joblib.load(BASE / 'models' / (crop.lower() + '_yield_et_rf.joblib')) for crop in CROPS}


def decode_solution(x, config=None):
    config = config or load_config()
    x = np.asarray(x, dtype=float)
    if x.shape != (138,):
        raise ValueError('Expected 138 decision variables (30 crop + 54 density + 54 fertilizer).')
    fields = config['fields']
    dry = [int(i) for i in fields['dryland_indices']]
    paddy = set(int(i) for i in fields['paddy_indices'])
    area = {int(i):v for i,v in fields['area_by_field_ha_as_coded'].items()}
    choice = np.round(x[:30]).astype(int)
    rows=[]
    for i in range(54):
        field_id=i+1
        crop='Rice' if field_id in paddy else ('Maize' if choice[dry.index(field_id)] == 1 else 'Soybean')
        dmin,dmax=config['bounds']['density_plants_m2'][crop]
        fmin,fmax=config['bounds']['fertilizer_kg_ha'][crop]
        rows.append({'Field_ID':field_id,'Area_ha':area[field_id],'Optimized_Crop':crop,
                     'Density_plants_m2': dmin+x[30+i]*(dmax-dmin),
                     'Fert_Weight_kg_ha': fmin+x[84+i]*(fmax-fmin)})
    return pd.DataFrame(rows)


def evaluate_plan(plan, models=None, config=None):
    config = config or load_config()
    models = models or load_models()
    p=plan.copy()
    required=['Field_ID','Area_ha','Optimized_Crop']+FEATURES
    missing=set(required)-set(p.columns)
    if missing:raise ValueError(f'Missing columns: {missing}')
    # Batch prediction by crop; same two RF inputs and two outputs as supplied optimizer.
    for name in ['Yield_ha','ET_mm','Profit_ha','Carbon_ha','WUE']:
        p[name]=np.nan
    ec=config['economic']
    for crop in CROPS:
        mask=p['Optimized_Crop']==crop
        if not mask.any():continue
        values=models[crop].predict(p.loc[mask,FEATURES])
        weight=p.loc[mask,'Fert_Weight_kg_ha'].to_numpy(dtype=float)
        pure_n=weight*0.15
        pure_p=pure_n*0.436
        pure_k=pure_n*0.830
        profit=(values[:,0]*ec['crop_price_cny_per_kg'][crop] - (ec['base_cost_cny_per_ha'][crop]+pure_n*ec['fert_price_per_kg_n_as_coded'][crop]))
        carbon=(ec['base_carbon_kg_co2e_per_ha'][crop]+pure_n*ec['n_fert_carbon_factor']+pure_p*ec['p_fert_carbon_factor']+pure_k*ec['k_fert_carbon_factor'])
        p.loc[mask,'Yield_ha']=values[:,0]
        p.loc[mask,'ET_mm']=values[:,1]
        p.loc[mask,'Profit_ha']=profit
        p.loc[mask,'Carbon_ha']=carbon
        p.loc[mask,'WUE']=values[:,0]/(values[:,1]*10)
    area=p['Area_ha'].to_numpy(dtype=float)
    summary={'Profit':float(np.sum(p['Profit_ha']*area)),
             'WUE':float(np.sum(p['Yield_ha']*area)/np.sum(p['ET_mm']*10*area)),
             'Carbon':float(np.sum(p['Carbon_ha']*area))}
    return p,summary


def check_constraints(plan,config=None):
    config=config or load_config()
    summed=plan.groupby('Optimized_Crop')['Area_ha'].sum()
    return {'maize_area_ha':float(summed.get('Maize',0)),
            'soybean_area_ha':float(summed.get('Soybean',0)),
            'meets_maize_min':bool(summed.get('Maize',0)>=config['constraints']['minimum_maize_area_ha_as_coded']),
            'meets_soybean_min':bool(summed.get('Soybean',0)>=config['constraints']['minimum_soybean_area_ha_as_coded'])}
