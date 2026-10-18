import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.farmopt.core import load_models,load_config,FEATURES,decode_solution,evaluate_plan,check_constraints

class ReproPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models=load_models()
        cls.data=pd.read_csv(ROOT/'data/source_outputs/ML_Training_Data.csv')
    def test_training_data(self):
        self.assertEqual(self.data.shape[0],2100)
        self.assertEqual(self.data.Crop.value_counts().to_dict(),{'Maize':700,'Soybean':700,'Rice':700})
    def test_models_two_outputs(self):
        for crop in ['Maize','Soybean','Rice']:
            x=self.data[self.data.Crop==crop][FEATURES].head(2)
            y=self.models[crop].predict(x)
            self.assertEqual(y.shape,(2,2))
            self.assertTrue(np.isfinite(y).all())
    def test_plan_reconstruction(self):
        p=pd.read_csv(ROOT/'data/source_outputs/Optimized_Farm_Plan_Continuous.csv')
        result,score=evaluate_plan(p,self.models)
        self.assertEqual(len(result),54)
        self.assertTrue(all(np.isfinite(z) for z in score.values()))
        check=check_constraints(p)
        self.assertTrue(check['meets_maize_min'] and check['meets_soybean_min'])
    def test_decoder(self):
        p=decode_solution(np.full(138,0.5))
        self.assertEqual(len(p),54)
        self.assertEqual(int(p.Field_ID.nunique()),54)
    def test_archive_front(self):
        f=pd.read_csv(ROOT/'data/source_outputs/Pareto_Front_Data.csv')
        self.assertEqual(len(f),100)
        self.assertEqual(list(f.columns),['Profit','WUE','Carbon'])
        self.assertTrue(np.isfinite(f.to_numpy()).all())

if __name__=='__main__': unittest.main()
