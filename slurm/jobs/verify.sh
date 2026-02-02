#!/bin/bash -l
#SBATCH -J region_verify
#SBATCH --time=0-04:00:00
#SBATCH -p largemem,batch,preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128g
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Verify
#

cd "$HOME/mcm2026"

module purge
module load miniforge/24.11.2-py312 2>/dev/null || module load miniforge/24.7.1-py312 2>/dev/null || module load miniforge 2>/dev/null
source activate ~/mcm2026/mcm_env

# With 16 workers, each gets 4 threads (16*4=64 CPUs)
export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4

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

source deactivate || true
exit $EXIT_CODE
