#!/bin/bash -l
#SBATCH -J region_finalize
#SBATCH --time=1-00:00:00
#SBATCH -p batch,preempt
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH --output=logs/finalize_%j.out
#SBATCH --error=logs/finalize_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Finalize
#

cd "$HOME/mcm2026"

module purge
module load anaconda/2021.05
source activate region-analysis

export NUMBA_NUM_THREADS=${SLURM_NTASKS:-8}
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}

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

conda deactivate
exit $EXIT_CODE
