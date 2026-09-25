import json,shutil,hashlib,numpy as np
from develop_phgeofuse_regression import OUT
selected=json.loads((OUT/'compact/frozen_candidate.json').read_text())['candidate'];bundle=OUT/'frozen_model';bundle.mkdir(exist_ok=True)
shutil.copy2(OUT/f"compact/{selected['name']}.joblib",bundle/'sequence.joblib')
shutil.copy2(OUT/f"retrieval_residual/model_weight{selected['residual_power']}_i150.joblib",bundle/'residual.joblib')
cfg={**selected,'training_mean_label':float(np.load(OUT/'development_features.npz')['y'].mean()),'dataset':'phopt','training_count':7124,'validation_count':760,'test_used_for_selection':False,'feature_provenance':json.loads((OUT/'esm2_masked/provenance.json').read_text()),'file_hashes':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in ['sequence.joblib','residual.joblib']}}
(bundle/'model.json').write_text(json.dumps(cfg,indent=2));print('FROZEN_BUNDLE',bundle)
