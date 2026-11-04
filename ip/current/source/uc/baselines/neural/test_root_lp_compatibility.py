"""Small solver API regression tests; no realistic instance optimization."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import neural_support
neural_support.enable_runtime()
import nd_root_lp_features as rootlp


def toy_model():
    return SimpleNamespace(A=sparse.csr_matrix([[1.,1e-15]]),
        row_lower=np.array([.5]),row_upper=np.array([np.inf]),
        col_lower=np.zeros(2),col_upper=np.ones(2),integrality=np.ones(2))


class RootLPCompatibilityTests(unittest.TestCase):
    def test_warning_retains_original_matrix_certificate(self):
        extractor=rootlp.RootLPFeatureExtractor(toy_model(),time_limit=5.)
        features,metadata=extractor.extract(np.array([1.,2.]))
        self.assertTrue(metadata["certified"])
        self.assertTrue(metadata["model_load_warning"])
        self.assertEqual(metadata["source_nonzeros"],2)
        self.assertEqual(metadata["loaded_nonzeros"],1)
        self.assertEqual(metadata["nonzeros_removed_on_model_load"],1)
        self.assertEqual(extractor.A[0,1],1e-15)
        self.assertAlmostEqual(metadata["primal_objective"],.5)
        self.assertLessEqual(metadata["continuous_feasibility_violation"],1e-6)
        self.assertLessEqual(metadata["dual_stationarity_residual"],metadata["dual_stationarity_tolerance"])
        self.assertEqual(features.shape,(3,15))
        self.assertTrue(np.isfinite(features).all())

    def test_model_load_error_is_rejected_before_solve(self):
        actual=rootlp.highspy.Highs
        class Rejected:
            def __init__(self):self.actual=actual()
            def __getattr__(self,name):return getattr(self.actual,name)
            def passModel(self,lp):return rootlp.highspy.HighsStatus.kError
            def run(self):raise AssertionError("Invalid model must not be solved")
        with patch.object(rootlp.highspy,"Highs",Rejected):
            with self.assertRaises(rootlp.RootLPFeatureError) as caught:
                rootlp.RootLPFeatureExtractor(toy_model()).extract(np.array([1.,2.]))
        self.assertEqual(caught.exception.metadata["model_load_status"],"HighsStatus.kError")

    def test_incomplete_loaded_dimensions_are_rejected(self):
        actual=rootlp.highspy.Highs
        class Incomplete:
            def __init__(self):self.actual=actual()
            def __getattr__(self,name):return getattr(self.actual,name)
            def getNumRow(self):return 0
            def run(self):raise AssertionError("Incomplete model must not be solved")
        with patch.object(rootlp.highspy,"Highs",Incomplete):
            with self.assertRaisesRegex(rootlp.RootLPFeatureError,"incomplete model dimensions"):
                rootlp.RootLPFeatureExtractor(toy_model()).extract(np.array([1.,2.]))


if __name__=="__main__":unittest.main()
