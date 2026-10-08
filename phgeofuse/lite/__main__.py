import argparse
import json

from phgeofuse.config import load_config
from .model import DESIGNS
from .pipeline import fit, predict


def main():
    parser = argparse.ArgumentParser(description="Fit or use PHGeoFuse-Lite on prepared SaProt inputs")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("fit", help="fit on train and select alpha on validation")
    training.add_argument("--config", required=True)
    training.add_argument("--output")
    training.add_argument("--design", choices=DESIGNS)
    for name in ("predict", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--manifest")
        command.add_argument("--split", default="predict" if name == "predict" else "validation")
        command.add_argument("--output", required=True)
        command.add_argument("--build-retrieval", action="store_true",
                             help="compute absent query rows in memory using the fixed reference library")
    args = parser.parse_args()
    config = load_config(args.config)
    result = (fit(config, output=args.output, design=args.design) if args.command == "fit"
              else predict(config, args.model, manifest=args.manifest, split=args.split,
                           output=args.output, evaluate=args.command == "evaluate",
                           build_queries=args.build_retrieval))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
