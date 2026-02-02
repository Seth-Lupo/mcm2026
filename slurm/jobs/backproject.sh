#!/bin/bash -l
#SBATCH -J region_backproj
#SBATCH --time=1-00:00:00
#SBATCH -p largemem,batch,preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256g
#SBATCH --output=logs/backproj_%j.out
#SBATCH --error=logs/backproj_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Backprojection
#

cd "$HOME/mcm2026"

module purge
module load miniforge/24.11.2-py312 2>/dev/null || module load miniforge/24.7.1-py312 2>/dev/null || module load miniforge 2>/dev/null
source activate ~/mcm2026/mcm_env

# With 16 workers, each gets 4 threads (16*4=64 CPUs)
export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4

echo "=============================================="
echo "Region Analysis - Backprojection"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_NTASKS"
echo "Config:     region-analysis/config.yaml"
echo "=============================================="

if [[ ! -f data/regions.json ]]; then
    echo "ERROR: data/regions.json not found!"
    exit 1
fi

python region-analysis/backprojection.py \
    --input data/regions.json \
    --output data/regions-backprojected.json
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

source deactivate || true
exit $EXIT_CODE
