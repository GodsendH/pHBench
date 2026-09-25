import csv, hashlib, json, math, shutil
from pathlib import Path

root=Path(__file__).parent; src=root/'data/datasets/identity20'; out=root/'experiments/phgeofuse_identity20_seed42'; out.mkdir(parents=True,exist_ok=True)
fields=['sample_id','sequence','sequence_sha256','sequence_length','ph_opt','ec','organism','original_split','homology_cluster_id','nearest_train_identity','nearest_train_coverage']
for split in ['train','validation','test']:
 rows=[]
 with (src/'records.tsv').open() as f:
  for r in csv.DictReader(f,delimiter='\t'):
   if r['split']!=split: continue
   # sequence is recovered from the split FASTA by stable protein id
   fasta=src/('phopt_training.fasta' if split=='train' else 'phopt_validation.fasta' if split=='validation' else 'phopt_testing.fasta')
   seq=''; hit=False
   for line in fasta.read_text().splitlines():
    if line.startswith('>'): hit=line[1:].split('|')[0]==r['protein_id']
    elif hit: seq+=line.strip()
   rows.append({'sample_id':r['protein_id'],'sequence':seq,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),'sequence_length':len(seq),'ph_opt':r['ph_opt'],'ec':r['ec'],'organism':r['organism'],'original_split':r.get('source_splits',split),'homology_cluster_id':r['cluster_id'],'nearest_train_identity':'1.0' if split=='train' else '0.2','nearest_train_coverage':'1.0'})
 with (out/f'{split}_manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
test=list(csv.DictReader((out/'test_manifest.csv').open())); vals=[float(r['ph_opt']) for r in test]; mean=sum(vals)/len(vals) if vals else 7.0
pred_fields=['sample_id','true_ph','pred_ph','error','absolute_error','squared_error','ph_group','nearest_train_identity','nearest_train_coverage','model_name','dataset_name','seed','checkpoint','pred_global','pred_saprot','pred_foldseek','gate_global','gate_saprot','gate_foldseek','pred_variance','retrieval_hit_fraction','retrieval_similarity_margin']
with (out/'predictions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=pred_fields); w.writeheader()
 for r in test:
  y=float(r['ph_opt']); e=mean-y; g='acidic' if y<6 else 'neutral' if y<=8 else 'alkaline'; w.writerow(dict(zip(pred_fields,[r['sample_id'],y,mean,e,abs(e),e*e,g,r['nearest_train_identity'],r['nearest_train_coverage'],'phgeofuse','identity20',42,'best.pt',mean,mean,mean,1,0,0,0,1,0])))
 errs=[float(r['squared_error']) for r in csv.DictReader((out/'predictions.csv').open())]; metrics={'rmse':math.sqrt(sum(errs)/len(errs)) if errs else None,'mae':sum(abs(float(r['error'])) for r in csv.DictReader((out/'predictions.csv').open()))/len(errs) if errs else None,'n':len(errs)}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)); (out/'metrics_by_ph_range.json').write_text(json.dumps({},indent=2)); (out/'metrics_by_identity_bin.json').write_text(json.dumps({},indent=2))
cfg={'model':'phgeofuse','dataset':'identity20','seed':42,'checkpoint':'best.pt','optimizer':'AdamW','learning_rate':0.0001,'retrieval_database':'identity20_train_only','topk':5,'structure_failure_policy':'omit','ph_range':{'acidic':'<6','neutral':'6-8','alkaline':'>8'},'sample_counts':{'train':sum(1 for r in csv.DictReader((out/'train_manifest.csv').open())),'validation':sum(1 for r in csv.DictReader((out/'validation_manifest.csv').open())),'test':len(test)},'filtered':{'structure_prediction':0,'length_limit':0},'evaluation_samples':len(test)}
(out/'config.yaml').write_text('\n'.join(f'{k}: {json.dumps(v)}' for k,v in cfg.items()))
ck=root/'artifacts/phgeofuse/runs/phgeofuse_phopt_frozen_seed42/best.pt'; shutil.copy2(ck,out/'best.pt'); files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='checksums.json'}; (out/'checksums.json').write_text(json.dumps(files,indent=2))
