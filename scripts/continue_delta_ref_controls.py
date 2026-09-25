"""Complete existing cheap controls, with no route to automatic LoRA training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from phgeofuse.config import path
    from phgeofuse.delta_ref.data import DevelopmentData
    from phgeofuse.delta_ref.experiment import (
        audit, evaluate_ablations, experiment_lock, status, write_report,
    )
    from phgeofuse.delta_ref.comparisons import nested_classical
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/delta_ref_phopt.yaml")
    args = parser.parse_args()
    data = DevelopmentData.load(args.config)
    output = path(data.config, "paths.output")
    with experiment_lock(output):
        audit(data, output)
        main_result = json.loads((output / "frozen/results.json").read_text())
        seed = data.config["protocol"]["development_seed"]
        status(output, "bounded_controls_started")
        nested_classical(data, output / "comparisons", seed)
        evaluate_ablations(data, output / "ablations", main_result, seed)
        status(output, "controls_complete_lora_paused",
               lora_triggered=main_result["lora_triggered_by_protocol"],
               local_lora_started=False, default_model_replaced=False,
               reason="User paused the DeltaRef LoRA route; no LoRA job is scheduled")
        write_report(output)


if __name__ == "__main__":
    main()
