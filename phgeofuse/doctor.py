from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import get, load_config, path


PACKAGES = {
    "torch": "2.0.0", "transformers": "4.40.0", "numpy": "1.26.1",
    "peft": None, "accelerate": None, "biotite": None, "propka": None,
    "faiss-cpu": None, "tenacity": None, "safetensors": None, "PyYAML": None,
}


def run_doctor(config: dict[str, Any]) -> dict[str, Any]:
    checks = []
    checks.append(_check("python", sys.version_info[:2] == (3, 10), platform.python_version(), expected="3.10.x"))
    checks.append(_check("conda_environment", bool(Path(sys.prefix).name == "phbench"), Path(sys.prefix).name, expected="phbench"))
    missing, drift = {}, {}
    for distribution, expected in PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing[distribution] = None
            continue
        if expected and not (actual == expected or actual.startswith(expected + "+")):
            drift[distribution] = {"expected": expected, "actual": actual}
    checks.append(_check("dependencies", not missing and not drift, "dependency audit", missing=missing, drift=drift))
    for name, configured in (
        ("foldseek", get(config, "structure.foldseek_binary", "foldseek")),
        ("mmseqs", get(config, "retrieval.mmseqs_binary", "mmseqs")),
        ("propka", get(config, "structure.propka_binary", "propka3")),
    ):
        binary = shutil.which(str(configured))
        if binary:
            result = subprocess.run([binary, "version"] if name != "propka" else [binary, "--version"], capture_output=True, text=True)
            version = (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else "available"
            checks.append(_check(name, result.returncode == 0, version, path=binary))
        else:
            checks.append(_check(name, False, "executable not found", configured=str(configured)))
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        checks.append(_check("cuda", gpu_ok, f"torch={torch.__version__}", devices=devices, bf16=torch.cuda.is_bf16_supported() if gpu_ok else False))
    except ImportError as exc:
        checks.append(_check("cuda", False, str(exc)))
    offline = bool(get(config, "runtime.offline", True))
    for name, repo, revision in (
        ("saprot_cache", get(config, "model.saprot_model"), get(config, "model.saprot_revision")),
        ("esmfold_cache", get(config, "structure.esmfold_model"), get(config, "structure.esmfold_revision")),
    ):
        complete, details = _huggingface_cache_complete(str(repo), revision)
        status = complete if offline else True
        checks.append(_check(name, status, "complete" if complete else "incomplete or absent", **details))
    for name, key in (("structures", "paths.structures"), ("graphs", "paths.graphs"), ("embeddings", "paths.embeddings")):
        directory = path(config, key)
        count = sum(1 for item in directory.rglob("*") if item.is_file()) if directory.exists() else 0
        parent = directory if directory.exists() else next(item for item in [directory.parent, Path(config["_root"])] if item.exists())
        checks.append(_check(name, parent.exists() and os_access_writable(parent), f"{count} cached files", path=str(directory)))
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def _huggingface_cache_complete(repo: str, revision: str | None):
    if not repo or repo == "None":
        return False, {"repo": repo}
    try:
        from transformers.utils.hub import cached_file

        options = {"revision": revision, "local_files_only": True, "_raise_exceptions_for_missing_entries": False}
        config_file = cached_file(repo, "config.json", **options)
        tokenizer_file = cached_file(repo, "tokenizer_config.json", **options)
        if not config_file:
            return False, {"repo": repo, "revision": revision, "missing": ["config.json"]}
        missing = []
        weights_found = False
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_file = cached_file(repo, index_name, **options)
            if index_file:
                weights_found = True
                index = json.loads(Path(index_file).read_text())
                for filename in sorted(set(index.get("weight_map", {}).values())):
                    if not cached_file(repo, filename, **options):
                        missing.append(filename)
        if not weights_found:
            weights_found = any(cached_file(repo, name, **options) for name in ("model.safetensors", "pytorch_model.bin"))
        if not weights_found:
            missing.append("model weights")
        if not tokenizer_file:
            missing.append("tokenizer_config.json")
        return not missing, {"repo": repo, "revision": revision, "missing": missing}
    except Exception as exc:
        return False, {"repo": repo, "revision": revision, "error": str(exc)}


def _check(name: str, ok: bool, message: str, **details):
    return {"name": name, "ok": bool(ok), "message": str(message), "details": details}


def os_access_writable(path: Path) -> bool:
    import os
    return os.access(path, os.W_OK)


def main():
    parser = argparse.ArgumentParser(description="Audit the pH-GeoFuse runtime")
    parser.add_argument("--config", default="configs/phgeofuse_phopt.yaml")
    args = parser.parse_args()
    report = run_doctor(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
