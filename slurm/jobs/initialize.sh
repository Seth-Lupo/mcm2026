#!/bin/bash -l
#SBATCH -J region_init
#SBATCH --time=2-00:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH --output=logs/init_%j.out
#SBATCH --error=logs/init_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Initialize (Sampling)
# Matches: python region-analysis/initialize.py --samples N --seed S --seasons X --output PATH
#

cd "$HOME/mcm2026"

# Load config
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Defaults (can be overridden in .env)
: "${N_SAMPLES:=20000000}"
: "${SEED:=67}"
: "${SEASONS:=}"
: "${TUFTS_EMAIL:=}"

# Set mail user if configured
if [[ -n "$TUFTS_EMAIL" ]]; then
    export SBATCH_MAIL_USER="$TUFTS_EMAIL"
fi

# Load modules - Tufts recommended setup
module purge
module load miniconda/23.10 2>/dev/null || module load anaconda/2021.05

# IMPORTANT: Tufts requires "source activate" not "conda activate"
source activate region-analysis

# Configure parallelism for numba
export NUMBA_NUM_THREADS=${SLURM_NTASKS:-16}
export OMP_NUM_THREADS=${SLURM_NTASKS:-16}
export MKL_NUM_THREADS=${SLURM_NTASKS:-16}

echo "=============================================="
echo "Region Analysis - Initialize"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_NTASKS"
echo "Memory:     ${SLURM_MEM_PER_NODE:-unknown}"
echo "Threads:    $NUMBA_NUM_THREADS"
echo "----------------------------------------------"
echo "Samples:    $N_SAMPLES"
echo "Seed:       $SEED"
echo "Seasons:    ${SEASONS:-all}"
echo "=============================================="

mkdir -p data logs

# Build command matching initialize.py argparse
CMD="python region-analysis/initialize.py"
CMD="$CMD --samples $N_SAMPLES"
CMD="$CMD --seed $SEED"
CMD="$CMD --output data/regions.json"

# Add seasons filter if specified
if [[ -n "$SEASONS" && "$SEASONS" != "all" ]]; then
    CMD="$CMD --seasons $SEASONS"
fi

echo ""
echo "Running: $CMD"
echo ""

$CMD
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

conda deactivate
exit $EXIT_CODE
