#!/bin/bash -l
#SBATCH -J region_pipeline
#SBATCH --time=4-00:00:00
#SBATCH -p largemem,batch,preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256g
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Full Pipeline (all steps in one job)
#

cd "$HOME/mcm2026"

module purge
module load miniforge/24.11.2-py312 2>/dev/null || module load miniforge/24.7.1-py312 2>/dev/null || module load miniforge 2>/dev/null
source activate ~/mcm2026/mcm_env

# With 32 workers, each gets 2 threads (32*2=64 CPUs)
export NUMBA_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "=============================================="
echo "Region Analysis - Full Pipeline"
echo "=============================================="
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo "Git status: $(git status --short | head -5)"
echo "Config hash: $(md5sum region-analysis/config.yaml | cut -d' ' -f1)"
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

source deactivate || true
exit $EXIT_CODE
