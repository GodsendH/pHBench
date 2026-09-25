import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, atomic_npz, write_predictions
from phgeofuse.delta_ref.model import DeltaNetwork, ReferencePredictor, Standardizer
from phgeofuse.delta_ref.training import fit_subset, save_bundle
from phgeofuse.delta_ref.inference import predict_cached
from phgeofuse.delta_ref.evaluation import compare_csvs, followup_test


def fixture():
    rng = np.random.default_rng(9)
    n = 75
    x = rng.normal(size=(n, 12)); y = np.tile([3., 7., 11.], n//3)
    keys = np.array([f'train::{i}' for i in range(n)])
    settings = {'device': 'cpu', 'threads': 2, 'batch_size': 128, 'learning_rate': .0003,
                'weight_decay': .001, 'max_epochs': 2, 'patience': 1, 'pairs_per_query': 8,
                'natural_partners': 4, 'gradient_clip': 1.}
    config = {'training': settings, 'model': {'widths': [8], 'frequency_powers': [0.],
              'dropout': .1, 'references_per_bin': 3}, 'protocol': {'bootstrap_draws': 100}}
    return DevelopmentData([], keys, x, x, y, keys.copy(), np.arange(n) % 5,
                           np.full(n, 'train'), config, {'synthetic_fixture': True})


class SyntheticBaseline:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def predict_excluded(self, excluded):
        fit, query = self.data.partition(excluded)
        assert not set(self.data.groups[fit]) & set(self.data.groups[query])
        self.calls.append(tuple(sorted(excluded)))
        b = np.full(len(query), np.mean(self.data.labels[fit]))
        return query, {'prediction': b, 'low_homology': np.ones(len(query), bool)}


class DeltaRefPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cached_inference_aligns_rows_and_rejects_labels_and_wrong_adapter(self):
        data = fixture()
        head = DeltaNetwork(12, 8, 0.)
        p = ReferencePredictor(head, Standardizer.fit(data.x[:60]), data.x[:9], data.labels[:9],
                               data.groups[:9], data.keys[:9], np.ones(9)/9)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_bundle(root/'model', p, {'kind': 'pair'}, 0., {})
            keys = data.keys[60:]
            atomic_npz(root/'x.npz', keys=keys, features=data.x[60:])
            b = np.linspace(5, 8, len(keys))
            write_predictions(root/'b.csv', keys[::-1], {'prediction': b[::-1]})
            out = predict_cached(root/'model', root/'x.npz', root/'b.csv', root/'out.csv')
            np.testing.assert_array_equal(out['prediction'], b)
            atomic_npz(root/'bad.npz', keys=keys, features=data.x[60:], labels=data.labels[60:])
            with self.assertRaisesRegex(ValueError, 'label-free'):
                predict_cached(root/'model', root/'bad.npz', root/'b.csv', root/'bad.csv')
            save_bundle(root/'lora', p, {'kind': 'pair', 'representation': 'lora'}, 0., {})
            with self.assertRaisesRegex(ValueError, 'fitted adapter'):
                predict_cached(root/'lora', root/'x.npz', root/'b.csv', root/'bad.csv')

    def test_evaluation_refuses_unaccepted_release_before_loading_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            atomic_json(Path(tmp)/'frozen_release.json', {'ready_for_followup_test': False})
            with patch('phgeofuse.delta_ref.evaluation.DevelopmentData.load') as loader:
                with self.assertRaisesRegex(ValueError, 'test access denied'):
                    followup_test('not_even_opened.yaml', tmp)
                loader.assert_not_called()

    def test_fixed_baseline_stop_preserves_original_scheduler_horizon(self):
        from tests.test_phgeofuse import _tiny_config, _tiny_records
        from phgeofuse.engine import train_model, _cosine_schedule
        from utils.distributed import DistributedContext
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);config=_tiny_config()
            config.update(_root=str(root),paths={'retrieval':str(root/'retrieval.pt'),'runs':str(root/'runs')},
                training={'seed':3,'run_name':'fixed_stop','per_device_batch_size':2,'global_batch_size':2,
                          'num_workers':0,'epochs':40,'stop_after_epochs':2,'learning_rate':.001,
                          'weight_decay':0.,'warmup_fraction':0.,'early_stopping_patience':41})
            config['retrieval'].update(top_k=1,require_foldseek=False,mmseqs_binary='missing')
            config['structure']={'foldseek_binary':'missing'}
            train_model(_tiny_records(root),config,DistributedContext(False,0,0,1,torch.device('cpu')))
            state=torch.load(root/'runs/fixed_stop_frozen_seed3/last.pt',map_location='cpu')
            self.assertEqual(state['epoch'],1)
            self.assertEqual(state['global_step'],2)
            self.assertAlmostEqual(state['optimizer_state_dict']['param_groups'][0]['lr'],.001*_cosine_schedule(2,0,40))

    def test_followup_evaluation_writes_all_seeds_and_keeps_field_gate_closed(self):
        data=fixture()
        data.keys=np.array([k.replace('train::','test::') for k in data.keys])
        data.groups=np.repeat([str(i) for i in range(25)],3)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);data.config['_root']=str(root)
            data.config['paths']={'source_experiment':str(root/'historical')}
            release={'models':[],'baseline_test_files':{}}
            for seed in (0,1,2,3,42):
                directory=root/f'final/seed{seed}'
                network=DeltaNetwork(12,8,0.)
                predictor=ReferencePredictor(network,Standardizer.fit(data.x),data.x[:9],data.labels[:9],
                    np.array([str(i) for i in range(9)]),np.array([f'train::{i}' for i in range(9)]),np.ones(9)/9)
                save_bundle(directory,predictor,{'kind':'pair'},0.,{})
                release['models'].append({'seed':seed,'path':str(directory)})
                source=root/f'historical/dual_test/seed{seed}.csv'
                write_predictions(source,data.keys,{'label':data.labels,'prediction':np.full(len(data.keys),7.)})
                release['baseline_test_files'][str(source)]=sha256_file(source)
            atomic_json(root/'frozen_release.json',release)
            def families(records,output):
                write_predictions(output,data.keys,{'group':data.groups})
                return data.groups
            with patch('phgeofuse.delta_ref.evaluation.verify_release',return_value=release),\
                 patch('phgeofuse.delta_ref.evaluation.load_config',return_value=data.config),\
                 patch('phgeofuse.delta_ref.evaluation.DevelopmentData.load',return_value=data),\
                 patch('phgeofuse.delta_ref.evaluation.sequence_families',side_effect=families),\
                 patch('phgeofuse.retrieval.RetrievalStore.load') as retrieval,\
                 patch('phgeofuse.delta_ref.comparisons.predict_controls',return_value={}):
                retrieval.return_value.features.return_value=torch.zeros(15)
                result=followup_test('fixture.yaml',root)
            self.assertFalse(result['field_leadership_established'])
            self.assertFalse(result['default_model_replaced'])
            self.assertEqual(len(result['missing_field_comparisons']),3)
            self.assertEqual(result['candidate']['mean']['all']['count'],75)
            with np.load(root/'followup_test/predictions.npz') as z:
                self.assertEqual(z['candidate'].shape,(5,75))

    def test_comparison_matches_seeds_labels_coverage_and_family_resamples(self):
        data = fixture(); keys = data.keys[:15]; y = data.labels[:15]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); candidate = []; baseline = []; partial = []
            write_predictions(root/'families.csv', keys, {'group': np.repeat(['a', 'b', 'c', 'd', 'e'], 3)})
            for seed, shift in zip((0, 1, 2, 3, 42), [-1., 1., -1., 1., 0.]):
                c = root/f'c{seed}.csv'; b = root/f'b{seed}.csv'; p = root/f'p{seed}.csv'
                write_predictions(c, keys, {'label': y, 'prediction': y+shift, 'seed': np.full(len(keys), seed)})
                write_predictions(b, keys[::-1], {'label': y[::-1], 'prediction': y[::-1]+2., 'seed': np.full(len(keys), seed)})
                write_predictions(p, keys[1:], {'label': y[1:], 'prediction': y[1:]+2., 'seed': np.full(len(keys)-1, seed)})
                candidate.append(c); baseline.append(b); partial.append(p)
            r = compare_csvs(candidate, {'baseline': baseline, 'partial': partial}, root/'families.csv', 100)
            self.assertAlmostEqual(r['candidate']['mean']['all']['rmse'], .8)
            self.assertEqual(r['coverage']['partial']['missing_acid'], 1)
            self.assertEqual(r['common_samples']['count'], 14)
            self.assertEqual(r['bootstrap']['unit'], 'family')
            with self.assertRaisesRegex(ValueError, 'seed order'):
                compare_csvs(candidate[::-1], {'baseline': baseline}, root/'families.csv', 100)
            write_predictions(baseline[0], keys, {'label': y+.01, 'prediction': y+2})
            with self.assertRaisesRegex(ValueError, 'label mismatch'):
                compare_csvs(candidate, {'baseline': baseline}, root/'families.csv', 100)

    def test_epoch_selection_cache_tracks_validation_labels(self):
        data = fixture(); fit, val = data.partition([0])
        recipe = {'name': 'pair_w8_p0', 'kind': 'pair', 'width': 8, 'power': 0.}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            fit_subset(data, fit, val, np.full(len(val), 7.), recipe, tmp, excluded=[0])
            changed = copy.deepcopy(data); changed.labels[val] += .01
            with self.assertRaisesRegex(ValueError, 'immutable'):
                fit_subset(changed, fit, val, np.full(len(val), 7.), recipe, tmp, excluded=[0])

    def test_nested_outer_labels_cannot_change_its_inner_selection_or_weights(self):
        from phgeofuse.delta_ref.experiment import nested
        data = fixture(); changed = copy.deepcopy(data)
        outer0 = data.folds == 0
        changed.labels[outer0] += np.where(data.labels[outer0] < 7, .1, -.1)
        recipe = [{'name': 'pair_w8_p0', 'kind': 'pair', 'width': 8, 'power': 0.}]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            provider = SyntheticBaseline(data)
            nested(data, root/'a', recipe, baseline=provider)
            nested(changed, root/'b', recipe, baseline=SyntheticBaseline(changed))
            self.assertEqual(len(provider.calls), 25)
            a = json.loads((root/'a/outer_results.json').read_text())[0]['winner']
            b = json.loads((root/'b/outer_results.json').read_text())[0]['winner']
            for key in ('recipe', 'epochs', 'strength', 'rank'):
                self.assertEqual(a[key], b[key])
            wa = torch.load(root/'a/outer0/pair_w8_p0/refit/weights.pt', map_location='cpu')
            wb = torch.load(root/'b/outer0/pair_w8_p0/refit/weights.pt', map_location='cpu')
            for key in wa['state_dict']:
                torch.testing.assert_close(wa['state_dict'][key], wb['state_dict'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
