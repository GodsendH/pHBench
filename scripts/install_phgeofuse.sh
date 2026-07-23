#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-phbench}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/envs/miniforge3}"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Conda initialization script was not found under ${CONDA_ROOT}." >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_ROOT}"

SNAPSHOT_DIR="${PROJECT_ROOT}/artifacts/phgeofuse/environment"
mkdir -p "${SNAPSHOT_DIR}"
conda list --explicit > "${SNAPSHOT_DIR}/${ENV_NAME}-before.txt"

python -m pip install --upgrade-strategy only-if-needed \
  -r "${PROJECT_ROOT}/requirements-geofuse.txt"

if ! command -v foldseek >/dev/null 2>&1 || ! command -v mmseqs >/dev/null 2>&1; then
  conda install --yes --freeze-installed -c conda-forge -c bioconda foldseek mmseqs2
fi

python - <<'PY'
import numpy
import torch
import transformers

expected = {
    "torch": (torch.__version__.split("+")[0], "2.0.0"),
    "transformers": (transformers.__version__, "4.40.0"),
    "numpy": (numpy.__version__, "1.26.1"),
}
drift = {name: values for name, values in expected.items() if values[0] != values[1]}
if drift:
    raise SystemExit(f"Core dependency drift after installation: {drift}")
print("Core versions preserved:", {name: values[0] for name, values in expected.items()})
PY

conda list --explicit > "${SNAPSHOT_DIR}/${ENV_NAME}-after.txt"
python -m phgeofuse.doctor --config "${PROJECT_ROOT}/configs/phgeofuse_phopt.yaml"
