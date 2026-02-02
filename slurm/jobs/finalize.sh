#!/bin/bash -l
#SBATCH -J region_finalize
#SBATCH --time=1-00:00:00
#SBATCH -p largemem,batch,preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256g
#SBATCH --output=logs/finalize_%j.out
#SBATCH --error=logs/finalize_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Finalize
#

cd "$HOME/mcm2026"

module purge
module load miniforge/24.11.2-py312 2>/dev/null || module load miniforge/24.7.1-py312 2>/dev/null || module load miniforge 2>/dev/null
source activate ~/mcm2026/mcm_env

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

echo "=============================================="
echo "Region Analysis - Finalize"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_NTASKS"
echo "Config:     region-analysis/config.yaml"
echo "=============================================="

if [[ ! -f data/regions-forwardprojected.json ]]; then
    echo "ERROR: data/regions-forwardprojected.json not found!"
    exit 1
fi

# Uses config.yaml for hull-samples, simplex-samples, seed, etc.
python region-analysis/finalize.py \
    --input data/regions-forwardprojected.json \
    --output data/regions-finalized.json
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

source deactivate || true
exit $EXIT_CODE
