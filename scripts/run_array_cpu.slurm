#!/usr/bin/bash
#SBATCH --job-name=bp-evolve-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=kamarthi.v@northeastern.edu
#SBATCH --mail-type=END,FAIL
#
# CPU array for CPU-only seed families (MiniRocket: numpy ROCKET + sklearn
# RidgeCV, NO torch/GPU). ANTIPATTERNS 11: Vignan submits by hand. ANTIPATTERNS
# 16: right-sized (8 BLAS-using cores, 24G for the ~577k x 2000 feature matrix).
#
#   sbatch --array=0-1 --export=ALL,MANIFEST=experiments/iter_0000/manifest_cpu.json \
#          scripts/run_array_cpu.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/SensingandInnovationLab/vignankamarthi/BP-Inference-EVOLVE}"
DATA_ROOT="${DATA_ROOT:-data/raw}"

if [[ -z "${MANIFEST:-}" ]]; then
    echo "FATAL: MANIFEST env var is required (path to manifest_cpu.json)" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs
source .venv/bin/activate

check_dep() {
    python -c "import $1" 2>/dev/null || {
        echo "Installing missing dependency: $2"
        pip install $2 || { echo "FATAL: pip install $2 failed"; exit 1; }
    }
}
check_dep numpy "numpy>=1.26,<2.0"
check_dep sklearn "scikit-learn>=1.4"
# NOTE: no torch check -- CPU families do not use it.

RUN_DIR="$(python -c "
import json
m = json.load(open('${MANIFEST}'))
print(m['experiments'][${SLURM_ARRAY_TASK_ID}]['run_dir'])")"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "FATAL: invalid array index ${SLURM_ARRAY_TASK_ID}. RUN_DIR=${RUN_DIR}" >&2
    exit 1
fi
if [[ -f "${RUN_DIR}/result.json" ]]; then
    echo "FATAL: ${RUN_DIR}/result.json already exists. Delete it or edit the manifest." >&2
    exit 1
fi

echo "CPU array task ${SLURM_ARRAY_TASK_ID}: ${RUN_DIR}/run.py at $(date) on $(hostname)"
# tee output into the run dir so the full log returns via git (run.log is tracked).
python -u "${RUN_DIR}/run.py" --run-dir "${RUN_DIR}" --data-root "${DATA_ROOT}" 2>&1 \
    | tee "${RUN_DIR}/run.log"
echo "CPU array task ${SLURM_ARRAY_TASK_ID} complete at $(date)"
