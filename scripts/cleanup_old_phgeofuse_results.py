import json, shutil, subprocess, sys
from pathlib import Path
root=Path('/home/hetianci/projects/Venus-DREAM').resolve()
report=root/'experiments/phgeofuse_phopt_full_20260913'
paths=[root/'artifacts/phgeofuse/runs',root/'artifacts/phgeofuse/smoke',root/'artifacts/phgeofuse/predictions']
paths+=list((root/'artifacts/phgeofuse/datasets').glob('*/runs'))
paths+=list((root/'artifacts/phgeofuse/datasets').glob('*/predictions'))
paths += [p for p in (root/'experiments').glob('phgeofuse*') if p != report]
paths += list((root/'artifacts/phgeofuse/completion_20260912').glob('*.before'))
removed=[]
for p in paths:
    resolved=p.resolve()
    if root not in resolved.parents or resolved == report: raise ValueError(str(p))
    if p.exists():
        removed.append(str(p.relative_to(root)))
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
(report/'cleanup.json').write_text(json.dumps(removed,indent=2))
print(json.dumps(removed,indent=2))
