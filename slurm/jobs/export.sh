#!/bin/bash -l
#SBATCH -J region_export
#SBATCH --time=0-01:00:00
#SBATCH -p batch,preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err
#SBATCH --mail-type=END,FAIL

#
# Region Analysis - Export to CSV/TXT
#

cd "$HOME/mcm2026"

module purge
module load miniforge/24.11.2-py312 2>/dev/null || module load miniforge/24.7.1-py312 2>/dev/null || module load miniforge 2>/dev/null
source activate ~/mcm2026/mcm_env

echo "=============================================="
echo "Region Analysis - Export"
echo "=============================================="
echo "Started:    $(date)"
echo "=============================================="

python region-analysis/export.py --data-dir data
EXIT_CODE=$?

echo ""
echo "Files created:"
ls -la data/*.csv data/*.txt 2>/dev/null || echo "  (none)"

echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=============================================="

source deactivate || true
exit $EXIT_CODE
