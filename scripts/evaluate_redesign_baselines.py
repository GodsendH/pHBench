import sys,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'experiments/phgeofuse_redesign_20260914'
for seed in [0,1,2,3]:
 source=root/f'experiments/phgeofuse_phopt_full_20260913/seed{seed}'
 checkpoint=next((source/'runs').glob('*homology_gate_v3*/best_calibrated.pt'))
 with (out/f'baseline_seed{seed}_validation.log').open('w') as log:
  subprocess.run([sys.executable,'-u','-m','phgeofuse.evaluate','--config',str(source/'v3.yaml'),'--dataset','phopt','--checkpoint',str(checkpoint),'--split','validation','--output',str(out/f'baseline_seed{seed}_validation.csv')],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BASELINE_VAL',seed,flush=True)
