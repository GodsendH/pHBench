"""Run complete-PHOPT v3 training/evaluation for five seeds with an audit trail."""
import copy, csv, hashlib, json, subprocess, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/phgeofuse_phopt_homology_gate_v3.yaml"
INITIAL = ROOT / "artifacts/phgeofuse/datasets/identity20/runs/phgeofuse_phopt_tuned_mse_v1_frozen_seed42/best.pt"
SEEDS = [0, 1, 2, 3, 42]

def main():
    ROOT.joinpath("experiments").mkdir(exist_ok=True)
    report = ROOT / "experiments" / ("phgeofuse_phopt_complete_seeds_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    report.mkdir()
    base = yaml.safe_load(SOURCE.read_text())
    manifest = ROOT / base["paths"]["manifest"]
    rows = list(csv.DictReader(manifest.open()))
    counts = dict(Counter(r["split"] for r in rows))
    assert len(rows) == 9855 and all(r["status"] == "ready" for r in rows)
    metadata = {"seeds": SEEDS, "dataset": "phopt", "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "sample_counts": counts, "initial_checkpoint": str(INITIAL), "initial_sha256": hashlib.sha256(INITIAL.read_bytes()).hexdigest(), "initial_dataset": "identity20 transfer initialization", "base_config": base}
    (report / "experiment.json").write_text(json.dumps(metadata, indent=2))
    results = []
    for seed in SEEDS:
        config = copy.deepcopy(base); config["training"]["seed"] = seed
        config_path = ROOT / "configs" / f"phgeofuse_complete_seed{seed}.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        run = ROOT / base["paths"]["runs"] / f'{base["training"]["run_name"]}_frozen_seed{seed}'
        common = ["--config", str(config_path), "--dataset", "phopt"]
        stages = [("train", ["--init-checkpoint", str(INITIAL)]), ("calibrate", ["--checkpoint", str(run / "best.pt")]), ("evaluate", ["--checkpoint", str(run / "best_calibrated.pt"), "--split", "test", "--output", str(run / "test_predictions_calibrated.csv")])]
        seed_record = {"seed": seed, "sample_counts": counts, "run": str(run), "stages": []}
        for stage, args in stages:
            command = [sys.executable, "-u", "-m", "phgeofuse." + stage, *common, *args]
            log = report / f"seed{seed}_{stage}.log"
            started = datetime.now().isoformat(); result = subprocess.run(command, stdout=log.open("w"), stderr=subprocess.STDOUT); ended = datetime.now().isoformat()
            entry = {"stage": stage, "command": command, "started": started, "ended": ended, "returncode": result.returncode, "log": str(log)}
            if stage == "train": entry["metrics_jsonl"] = str(run / "metrics.jsonl")
            if stage == "evaluate":
                mp = run / "test_predictions_calibrated.metrics.json"; entry["metrics_file"] = str(mp); entry["metrics"] = json.loads(mp.read_text()) if mp.exists() else None
            seed_record["stages"].append(entry); (report / "progress.json").write_text(json.dumps({"completed_seed": seed, "stage": stage, "records": seed_record}, indent=2))
            if result.returncode: raise RuntimeError(f"seed {seed} {stage} failed; see {log}")
        results.append(seed_record); (report / "results.json").write_text(json.dumps(results, indent=2))
    (report / "status.json").write_text(json.dumps({"status": "complete", "seeds": SEEDS, "sample_counts": counts}, indent=2))
    print(f"Completed: {report}", flush=True)

if __name__ == "__main__": main()
