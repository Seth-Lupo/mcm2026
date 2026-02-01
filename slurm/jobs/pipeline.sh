#!/bin/bash -l
#SBATCH -J region_pipeline
#SBATCH --time=4-00:00:00
#SBATCH -p batch,preempt
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Full Pipeline (all steps in one job)
#

cd "$HOME/mcm2026"

module purge
module load anaconda/2021.05
source activate region-analysis

export NUMBA_NUM_THREADS=${SLURM_NTASKS:-16}
export OMP_NUM_THREADS=${SLURM_NTASKS:-16}
export MKL_NUM_THREADS=${SLURM_NTASKS:-16}

echo "=============================================="
echo "Region Analysis - Full Pipeline"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "CPUs:       $SLURM_NTASKS"
echo "Threads:    $NUMBA_NUM_THREADS"
echo "Config:     region-analysis/config.yaml"
echo "=============================================="

mkdir -p data logs

# Uses config.yaml for all settings
python region-analysis/main.py
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

conda deactivate
exit $EXIT_CODE
