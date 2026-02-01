#!/bin/bash -l
#SBATCH -J region_backproj
#SBATCH --time=1-00:00:00
#SBATCH -p batch,preempt
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH --output=logs/backproj_%j.out
#SBATCH --error=logs/backproj_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Backprojection
#

cd "$HOME/mcm2026"

module purge
module load anaconda/2021.05
source activate region-analysis

export NUMBA_NUM_THREADS=${SLURM_NTASKS:-8}
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}

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

conda deactivate
exit $EXIT_CODE
