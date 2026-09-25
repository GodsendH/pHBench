import sys,csv,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT,OUT,metrics
from phgeofuse.robust_fusion import RobustFusion
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key
records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split=='validation'];keys=[record_key(r) for r in records];F=np.load(OUT/'esm2_masked/features.npz');order={str(k):i for i,k in enumerate(F['keys'])};ix=[order[k] for k in keys];rows={r['key']:r for r in csv.DictReader((OUT/'baseline_validation.csv').open())};base=np.array([float(rows[k]['prediction']) for k in keys]);store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt');retrieval=np.array([store.features(k).numpy() for k in keys]);args=(base,F['mean'][ix],F['std'][ix],retrieval,[r.sequence for r in records]);reference=RobustFusion(OUT/'frozen_model').predict(*args)['prediction'];actual=RobustFusion(OUT/'refit_verification').predict(*args)['prediction'];np.testing.assert_allclose(reference,actual,atol=1e-7,rtol=0)
(OUT/'refit_verification/result.json').write_text(json.dumps({'passed':True,'max_absolute_prediction_difference':float(np.max(abs(reference-actual))),'validation_count':len(keys)},indent=2));print('REFIT_REPRODUCED',float(np.max(abs(reference-actual))))

