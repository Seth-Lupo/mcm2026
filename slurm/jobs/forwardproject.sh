#!/bin/bash -l
#SBATCH -J region_fwdproj
#SBATCH --time=1-00:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH --output=logs/fwdproj_%j.out
#SBATCH --error=logs/fwdproj_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Forward Projection
# Matches: python region-analysis/forwardprojection.py --input PATH --output PATH
#

cd "$HOME/mcm2026"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

module purge
module load miniconda/23.10 2>/dev/null || module load anaconda/2021.05
source activate region-analysis

export NUMBA_NUM_THREADS=${SLURM_NTASKS:-8}
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}

echo "=============================================="
echo "Region Analysis - Forward Projection"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_NTASKS"
echo "=============================================="

# Verify input exists
if [[ ! -f data/regions-backprojected.json ]]; then
    echo "ERROR: data/regions-backprojected.json not found!"
    exit 1
fi

python region-analysis/forwardprojection.py \
    --input data/regions-backprojected.json \
    --output data/regions-forwardprojected.json

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

conda deactivate
exit $EXIT_CODE
