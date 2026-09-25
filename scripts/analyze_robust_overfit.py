import csv,json,numpy as np
from develop_phgeofuse_regression import OUT,ROOT,metrics
from phgeofuse.robust_fusion import RobustFusion
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore,record_key
D=np.load(OUT/'development_features.npz');F=np.load(OUT/'esm2_masked/features.npz');records=[r for r in read_manifest(ROOT/'artifacts/phgeofuse/manifest.csv') if r.split=='train'];keys=[record_key(r) for r in records];rows={r['key']:r for r in csv.DictReader((OUT/'baseline_train.csv').open())};base=np.array([float(rows[k]['prediction']) for k in keys]);store=RetrievalStore.load(ROOT/'artifacts/phgeofuse/retrieval.pt');retrieval=np.array([store.features(k).numpy() for k in keys]);pred=RobustFusion(OUT/'frozen_model').predict(base,F['mean'][:7124],F['std'][:7124],retrieval,[r.sequence for r in records])
y=D['y'];btrain=float(np.sqrt(np.mean((base-y)**2)));ntrain=float(np.sqrt(np.mean((pred['prediction']-y)**2)));bval=json.loads((OUT/'baseline_validation.metrics.json').read_text())['rmse'];nval=json.loads((OUT/'frozen_model/model.json').read_text())['validation']['rmse']
r={'seed':42,'baseline_train_rmse':btrain,'baseline_validation_rmse':bval,'baseline_gap':bval-btrain,'candidate_train_rmse':ntrain,'candidate_validation_rmse':nval,'candidate_gap':nval-ntrain,'note':'in-sample training errors; not OOF. Smaller positive gap is a limited overfit diagnostic, not proof of no overfitting.'};(OUT/'overfit_diagnostic.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
