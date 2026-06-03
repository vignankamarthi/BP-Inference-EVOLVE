#!/usr/bin/env bash
#SBATCH --job-name=bp-evolve-submission
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=kamarthi.v@northeastern.edu
#SBATCH --mail-type=END,FAIL

# Single-task SLURM script for a HIP-G compliance/test-set evaluation. Mirrors
# run_array.slurm's environment so the submission runs in the SAME environment
# as every iteration job. It trains a champion + evaluates the blinded test
# split (CalFree / CalBased). Must NOT run on a login node.
#
# Submit from the project root on Explorer:
#   cd /projects/SensingandInnovationLab/vignankamarthi/BP-Inference-EVOLVE
#   sbatch --export=ALL,SUBMISSION_DIR=experiments/submission_01 \
#          scripts/run_submission.slurm

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/projects/SensingandInnovationLab/vignankamarthi/BP-Inference-EVOLVE}"
DATA_ROOT="${DATA_ROOT:-data/raw}"

if [[ -z "${SUBMISSION_DIR:-}" ]]; then
    echo "FATAL: SUBMISSION_DIR env var is required (path to experiments/submission_NN)" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p logs

if [[ ! -d "${SUBMISSION_DIR}" || ! -f "${SUBMISSION_DIR}/run.py" ]]; then
    echo "FATAL: ${SUBMISSION_DIR}/run.py not found" >&2
    exit 1
fi

source .venv/bin/activate

check_dep() {
    local import_name="$1"
    local install_spec="$2"
    python -c "import ${import_name}" 2>/dev/null || {
        echo "Installing missing dependency: ${install_spec}"
        pip install ${install_spec} || { echo "FATAL: pip install ${install_spec} failed"; exit 1; }
    }
}

check_dep numpy "numpy>=1.26"
check_dep scipy "scipy>=1.12"
check_dep sklearn "scikit-learn>=1.4"
check_dep h5py "h5py>=3.10"
check_dep pyarrow "pyarrow>=15.0"
check_dep tqdm "tqdm>=4.66"
check_dep torch "torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121"

CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_OK" != "True" ]]; then
    echo "WARNING: CUDA unavailable on $(hostname). Run will use CPU (slower)."
else
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo "Submission: running ${SUBMISSION_DIR}/run.py at $(date) on $(hostname)"
python -u "${SUBMISSION_DIR}/run.py" --run-dir "${SUBMISSION_DIR}" --data-root "${DATA_ROOT}"
echo "Submission ${SUBMISSION_DIR} complete at $(date)"
