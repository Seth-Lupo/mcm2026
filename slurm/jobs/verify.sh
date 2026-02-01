#!/bin/bash -l
#SBATCH -J region_verify
#SBATCH --time=0-04:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Verify (Check centroids/vertices against elimination constraints)
# CLI: python region-analysis/verify.py --regions PATH --events PATH --max-vertices N
#

cd "$HOME/mcm2026"

module purge
module load miniconda/23.10 2>/dev/null || module load anaconda/2021.05
source activate region-analysis

export NUMBA_NUM_THREADS=${SLURM_NTASKS:-8}
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}

echo "=============================================="
echo "Region Analysis - Verify"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_NTASKS"
echo "Config:     region-analysis/config.yaml"
echo "=============================================="

if [[ ! -f data/regions-finalized.json ]]; then
    echo "ERROR: data/regions-finalized.json not found!"
    exit 1
fi

if [[ ! -f data/events.json ]]; then
    echo "ERROR: data/events.json not found!"
    exit 1
fi

# Uses config.yaml for max-vertices, etc.
python region-analysis/verify.py \
    --regions data/regions-finalized.json \
    --events data/events.json
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

conda deactivate
exit $EXIT_CODE
