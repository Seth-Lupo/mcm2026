#!/bin/bash -l
#SBATCH -J region_init
#SBATCH --time=2-00:00:00
#SBATCH -p batch,preempt
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH --output=logs/init_%j.out
#SBATCH --error=logs/init_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Initialize (Sampling + Hull)
#

cd "$HOME/mcm2026"

module purge
module load anaconda/2021.05
source activate region-analysis

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
echo "Threads:    $NUMBA_NUM_THREADS"
echo "Config:     region-analysis/config.yaml"
echo "=============================================="

mkdir -p data logs

# Uses config.yaml for samples, seed, etc.
python region-analysis/initialize.py --output data/regions.json
EXIT_CODE=$?
echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

conda deactivate
exit $EXIT_CODE
