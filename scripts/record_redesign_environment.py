import sys,json,hashlib,importlib.metadata
from pathlib import Path
from transformers.utils.hub import cached_file
root=Path('/home/hetianci/projects/Venus-DREAM');out=root/'experiments/phgeofuse_redesign_20260914'
p=Path(cached_file('facebook/esm2_t33_650M_UR50D','config.json',local_files_only=True))
weight=Path(cached_file('facebook/esm2_t33_650M_UR50D','model.safetensors',local_files_only=True))
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):d.update(block)
 return d.hexdigest()
r={'python':sys.executable,'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','numpy','scipy','scikit-learn','biopython','joblib']},'esm2_revision':p.parent.name,'config_sha256':sha(p),'weights_sha256':sha(weight),'weights_path':str(weight),'weights_mtime':weight.stat().st_mtime}
(out/'environment_and_feature_revision.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

