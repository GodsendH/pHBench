import subprocess
from pathlib import Path
root=Path('/home/hetianci/projects/Venus-DREAM')
report=root/'experiments/phgeofuse_phopt_full_20260913'
command=['bash','-lc','source /home/hetianci/envs/miniforge3/etc/profile.d/conda.sh && conda activate phbench && exec python -u scripts/run_phgeofuse_phopt_full.py --report experiments/phgeofuse_phopt_full_20260913']
with (report/'runner.log').open('a') as log:
    process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(report/'runner.pid').write_text(str(process.pid))
print(process.pid)
