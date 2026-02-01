#!/bin/bash -l
#SBATCH -J region_zip
#SBATCH --time=0-00:30:00
#SBATCH -p batch,preempt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH --output=logs/transfer_%j.out
#SBATCH --error=logs/transfer_%j.err
#SBATCH --mail-type=END

#
# Package results into a zip for easy download
#

cd "$HOME/mcm2026"

echo "=============================================="
echo "Region Analysis - Package Results"
echo "=============================================="
echo "Started:    $(date)"
echo "=============================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="results_${TIMESTAMP}.zip"

echo "Creating: $ARCHIVE"
echo ""

zip -r "$ARCHIVE" \
    data/*.json \
    data/*.csv \
    data/*.txt \
    logs/*.out \
    logs/*.err \
    2>/dev/null

echo ""
echo "Archive contents:"
unzip -l "$ARCHIVE" | head -30

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo ""
echo "----------------------------------------------"
echo "Archive:    $HOME/mcm2026/$ARCHIVE"
echo "Size:       $SIZE"
echo "----------------------------------------------"
echo ""
echo "Download with:"
echo "  ./slurm/start.sh download"
echo ""
echo "Or manually:"
echo "  scp ${USER}@login.pax.tufts.edu:~/mcm2026/$ARCHIVE ./"
echo ""
echo "=============================================="
echo "Finished:   $(date)"
echo "=============================================="
