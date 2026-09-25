"""Run the PHOPT v3 seed comparison with the existing training recipe."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/phgeofuse_phopt_homology_gate_v3.yaml"
INITIAL = ROOT / "artifacts/phgeofuse/runs/phgeofuse_phopt_tuned_mse_v1_frozen_seed42/best.pt"
REPORT = ROOT / "experiments" / ("phgeofuse_v3_phopt_seeds_0_3_" + datetime.now().strftime("%Y%m%d_%H%M%S"))


def main():
    os.chdir(ROOT)
    REPORT.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(SOURCE.read_text())
    runs = [ROOT / base["paths"]["runs"] / f'{base["training"]["run_name"]}_frozen_seed{seed}' for seed in range(4)]
    if any(run.exists() for run in runs):
        raise FileExistsError("Seed output directory already exists; refusing to overwrite")
    metadata = {
        "python": sys.executable,
        "initial_checkpoint": str(INITIAL),
        "initial_sha256": hashlib.sha256(INITIAL.read_bytes()).hexdigest(),
        "base_config": base,
        "seeds": [0, 1, 2, 3],
    }
    (REPORT / "experiment.json").write_text(json.dumps(metadata, indent=2))
    results = []
    for seed, run in enumerate(runs):
        config = copy.deepcopy(base)
        config["training"]["seed"] = seed
        check = copy.deepcopy(config)
        check["training"]["seed"] = base["training"]["seed"]
        assert check == base
        config_path = ROOT / "configs" / f"phgeofuse_v3_phopt_seed{seed}.yaml"
        if config_path.exists():
            raise FileExistsError(config_path)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        common = ["--config", str(config_path), "--dataset", "phopt"]
        stages = [
            ("train", ["--init-checkpoint", str(INITIAL)]),
            ("calibrate", ["--checkpoint", str(run / "best.pt")]),
            ("evaluate", ["--checkpoint", str(run / "best_calibrated.pt"), "--split", "test", "--output", str(run / "test_predictions_calibrated.csv")]),
        ]
        for stage, args in stages:
            command = [sys.executable, "-u", "-m", "phgeofuse." + stage, *common, *args]
            status = {"seed": seed, "stage": stage, "command": command, "time": datetime.now().isoformat()}
            (REPORT / "status.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            with (REPORT / f"seed{seed}_{stage}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"seed {seed} {stage} failed: see {log.name}")
        metrics = json.loads((run / "test_predictions_calibrated.metrics.json").read_text())
        results.append({"seed": seed, **metrics})
        (REPORT / "results.json").write_text(json.dumps(results, indent=2))
        scalar_keys = [key for key, value in results[0].items() if isinstance(value, (int, float, str))]
        with (REPORT / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    (REPORT / "status.json").write_text(json.dumps({"status": "complete", "seeds": [0, 1, 2, 3]}, indent=2))
    print(f"Completed: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
