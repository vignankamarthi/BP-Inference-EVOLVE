#!/usr/bin/bash
#SBATCH --job-name=bp_build_cache
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu
#
# Build the PPG-only PulseDB split cache (CPU-bound HDF5 reads, no GPU).
# Writes data/raw/{train,validation,calfree,calbased,aami_cal,aami_test}.npz.
# Resumable: re-running skips splits whose .npz already exists.
#
# EDIT THIS to point at the raw PulseDB on the cluster (MIMIC/, VitalDB/, Info_Files/):
PULSEDB_DIR="${PULSEDB_DIR:-/projects/SensingandInnovationLab/vignankamarthi/PulseDB}"

PROJECT_DIR=/projects/SensingandInnovationLab/vignankamarthi/BP-Inference-EVOLVE
cd "$PROJECT_DIR" || exit 1
source .venv/bin/activate
mkdir -p logs data/raw

# ─── Dependency check: install if missing, die if install fails ───
check_dep() {
    python -c "import $1" 2>/dev/null || {
        echo "Installing missing dependency: $2"
        pip install "$2" || { echo "FATAL: failed to install $2"; exit 1; }
    }
}
check_dep numpy "numpy>=1.26,<2.0"
check_dep h5py "h5py>=3.10"

echo "PulseDB source: $PULSEDB_DIR"
python3 scripts/build_pulsedb_cache.py \
    --pulsedb-dir "$PULSEDB_DIR" \
    --out-dir "$PROJECT_DIR/data/raw" \
    --val-fraction 0.2 \
    --seed 0
