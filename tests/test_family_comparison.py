import unittest
import numpy as np
from localph.family_comparison import family_comparisons


class FamilyComparisonTests(unittest.TestCase):
    def test_paired_small_improvement_can_be_detected_without_fixed_noise_floor(self):
        y=np.linspace(10,12,22); baseline=y+np.linspace(.5,3,22)
        candidate=y+.99*(baseline-y)
        r=family_comparisons(y,{'new':candidate[None],'old':baseline[None]},np.arange(22),[('new','old')],draws=1000)
        alk=r['comparisons']['new']['old']['alkaline']['rmse']
        self.assertLess(abs(alk['point_difference']),.03)
        self.assertLess(alk['simultaneous_ci95'][1],0)

    def test_identical_predictions_have_exact_zero_core_difference(self):
        y=np.array([3.,7.,8.,11.,7.,8.]); p=(y+1)[None]
        r=family_comparisons(y,{'new':p,'old':p},[0,0,1,1,2,2],[('new','old')],draws=500)
        core=r['comparisons']['new']['old']['core']['rmse']
        self.assertEqual(r['family_count'],3)
        self.assertEqual(core['simultaneous_ci95'],[0.,0.])
        self.assertTrue(core['ci_upper_at_most_zero'])
        self.assertFalse(core['simultaneous_improvement'])

    def test_mean_seed_metrics_is_not_an_ensemble_metric(self):
        y=np.full(12,7.)
        baseline=np.stack([y+2,y-2]); candidate=np.stack([y+1,y-1])
        r=family_comparisons(y,{'new':candidate,'old':baseline},np.arange(12),[('new','old')],draws=100)
        self.assertEqual(r['comparisons']['new']['old']['core']['rmse']['point_difference'],-1.)
        self.assertEqual(r['comparisons']['new']['old']['acid']['rmse']['valid_draws'],0)

    def test_single_tail_family_does_not_silently_drop_empty_draws(self):
        y=np.r_[11.,np.full(19,7.)]
        r=family_comparisons(y,{'new':y[None],'old':(y+1)[None]},np.arange(20),[('new','old')],draws=1000)
        alk=r['comparisons']['new']['old']['alkaline']['rmse']
        self.assertGreater(alk['empty_endpoint_draws'],50)
        self.assertIsNone(alk['ci95'])


if __name__=='__main__': unittest.main()
