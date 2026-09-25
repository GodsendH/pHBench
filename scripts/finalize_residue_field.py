"""Wait for the bounded training supervisor, then verify and report artifacts.

This task does not launch training, retry failed training, or alter any model.
Run under run_bounded_local.py with a limit below 10 hours.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file


def main():
    base = ROOT / "experiments/residue_field_phopt_20260917"
    budget_path = base / "nested_v2_budget.json"
    status_path = base / "finalization_status.json"
    start = time.monotonic()
    source_names = ["scripts/report_residue_field.py", "scripts/check_residue_field_artifacts.py", str(Path(__file__).relative_to(ROOT))]
    sources = {p: sha256_file(ROOT / p) for p in source_names}
    atomic_json(status_path, {"state": "waiting_for_training", "source_hashes": sources})
    while True:
        budget = json.loads(budget_path.read_text())
        if budget["state"] != "running":
            break
        if time.monotonic() - start > 8.5 * 3600:
            raise TimeoutError("training supervisor did not finish within followup budget")
        time.sleep(20)
    if budget["state"] != "complete" or budget["returncode"] != 0:
        atomic_json(status_path, {"state": "training_not_successful", "training_budget": budget})
        raise RuntimeError("training failed or reached its budget; no automatic restart")
    for filename, digest in sources.items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError("finalization source changed while waiting")
    atomic_json(status_path, {"state": "verifying", "source_hashes": sources})
    for kind in ("direct", "global", "sparse"):
        subprocess.run([sys.executable, "-B", "scripts/check_residue_field_artifacts.py",
                        "--experiment", str(base / "nested_v2"),
                        "--features", str(base / "features/tokens.npz"),
                        "--fit", str(base / "nested_v2/outer0" / kind / "refit"),
                        "--output", str(base / f"reload_verification_final_{kind}.json")],
                       cwd=ROOT, check=True, timeout=600)
    subprocess.run([sys.executable, "-B", "scripts/report_residue_field.py",
                    "--experiment", str(base / "nested_v2"), "--budget", str(budget_path),
                    "--output", str(base / "analysis")], cwd=ROOT, check=True, timeout=600)
    atomic_json(status_path, {"state": "complete", "source_hashes": sources,
                             "elapsed_seconds": time.monotonic() - start,
                             "goal_achieved": False, "default_model_replaced": False,
                             "report": str(base / "analysis/REPORT_ZH.md")})


if __name__ == "__main__":
    main()
