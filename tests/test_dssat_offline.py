import importlib.util
from pathlib import Path
import sys, unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dssat_renderer',ROOT/'scripts/00_prepare_dssat_examples.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DSSATOfflineTests(unittest.TestCase):
    def test_templates_can_be_rendered_without_dssat(self):
        for crop,(tpl,out) in module.TEMPLATES.items():
            src=(ROOT/'dssat/templates'/tpl).read_bytes()
            self.assertIn(b'\r\n',src)
            result=module.render(src,crop=crop,soil_id='XY00000001',density=15.1,fertilizer=2000)
            self.assertEqual(len(src),len(result))
            self.assertEqual(src.count(b'\r\n'),result.count(b'\r\n'))
            for token in module.TOKENS:self.assertNotIn(token,result)
    def test_bad_soil_code_rejected(self):
        tpl=(ROOT/'dssat/templates/TPL_MAIZE.MZX').read_bytes()
        with self.assertRaises(ValueError):
            module.render(tpl,crop='Maize',soil_id='INVALID',density=9.0,fertilizer=1300)
    def test_numeric_conversions_match_batch_source(self):
        self.assertEqual(module.convert_fertilizer(2000),(300,131,249))

if __name__=='__main__':unittest.main()
