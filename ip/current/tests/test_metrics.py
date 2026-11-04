import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('current_verify',Path(__file__).resolve().parents[1]/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class Metrics(unittest.TestCase):
    def data(self,p):
        return [dict(problem=p,n=32,method='test',seed=1,index=i,K=4,anchor=0.,reference=10. if i else 100.,value=5. if i else 100.,valid=True,recovery=i==0) for i in range({'uc':256,'grouped':360,'correlated':300}[p])]
    def fits(self,p):return [dict(problem=p,n=32,method='test',fit_seconds=[1.,3.])]
    def test_ratio_of_sums(self):
        r=v.summarize(self.data('correlated'),self.fits('correlated'))['correlated',32,'test']
        self.assertAlmostEqual(r['objective_ratio'],(100+299*5)/(100+299*10));self.assertEqual(r['learning_fit_seconds'],2.)
    def test_minimization_signs(self):
        d=self.data('uc');a=v.summarize(d,self.fits('uc'))['uc',32,'test']
        for q in d:
            for k in ['anchor','reference','value']:q[k]=-q[k]
        b=v.summarize(d,self.fits('uc'))['uc',32,'test'];self.assertEqual(a['objective_ratio'],b['objective_ratio'])
    def test_grouped_failure_denominator(self):
        d=self.data('grouped');d[0].update(valid=False,recovery=False,value=None)
        r=v.summarize(d,self.fits('grouped'))['grouped',32,'test']
        self.assertEqual(r['recovery_percent'],0.);self.assertEqual(r['planned_tests'],360)
        self.assertEqual(r['valid_ratio_tests'],359);self.assertEqual(r['objective_ratio'],.5)
    def test_duplicate_index(self):
        d=self.data('correlated');d[-1]['index']=0
        with self.assertRaises(ValueError):v.summarize(d,self.fits('correlated'))
    def test_incomplete_relative_cohort(self):
        d=self.data('uc');d[0].update(valid=False,recovery=False,value=None)
        with self.assertRaises(ValueError):v.summarize(d,self.fits('uc'))
    def test_zero_denominator(self):
        d=self.data('correlated')
        for q in d:q['anchor']=q['reference']
        with self.assertRaises(ValueError):v.summarize(d,self.fits('correlated'))
if __name__=='__main__':unittest.main()
