#!/bin/bash -l
#SBATCH -J region_export
#SBATCH --time=0-01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err

#
# Region Analysis - Export to CSV/TXT
# Matches: python region-analysis/export.py --data-dir PATH
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

conda deactivate
exit $EXIT_CODE
