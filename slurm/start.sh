#!/bin/bash
#
# SLURM Pipeline Orchestrator for Region Analysis
#
# Usage:
#   ./slurm/start.sh [command]
#
# Commands:
#   setup      - Clone repo and setup environment on cluster
#   run        - Submit full pipeline (init -> backproj -> fwdproj -> export -> zip)
#   status     - Check job status
#   download   - Download results to local machine
#   logs       - View recent job logs
#   clean      - Cancel all your jobs
#   ssh        - Open SSH session to cluster
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
else
    echo "ERROR: .env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

CLUSTER_HOST="login.cluster.tufts.edu"
CLUSTER_SSH="${TUFTS_USER}@${CLUSTER_HOST}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

#
# SETUP: Clone repo and create conda environment on cluster
#
cmd_setup() {
    log_info "Setting up environment on Tufts HPC..."

    # First, upload .env so it's available on cluster
    log_info "Uploading configuration..."
    ssh "$CLUSTER_SSH" "mkdir -p ~/mcm2026"
    scp "$PROJECT_DIR/.env" "${CLUSTER_SSH}:~/mcm2026/.env"

    ssh "$CLUSTER_SSH" 'bash -s' << 'REMOTE_SETUP'
        set -e
        cd ~

        # Load .env if exists
        if [[ -f ~/mcm2026/.env ]]; then
            set -a
            source ~/mcm2026/.env
            set +a
        fi

        # Use miniconda (recommended by Tufts) or anaconda
        module purge
        module load miniconda/23.10 2>/dev/null || module load anaconda/2021.05

        # Clone or update repo
        REPO_URL="${GITHUB_REPO:-https://github.com/your-username/mcm2026.git}"
        if [[ -d ~/mcm2026/.git ]]; then
            echo "Repository exists, pulling latest..."
            cd ~/mcm2026
            git pull || echo "Git pull failed, continuing with existing code"
        else
            echo "Cloning repository..."
            rm -rf ~/mcm2026
            git clone "$REPO_URL" ~/mcm2026
        fi
        cd ~/mcm2026

        # Create conda environment if needed
        if ! conda env list | grep -q "^region-analysis "; then
            echo "Creating conda environment with Python 3.11..."
            conda create -n region-analysis python=3.11 pip -y
        fi

        # Activate and install deps
        # IMPORTANT: Tufts requires "source activate" not "conda activate"
        source activate region-analysis

        echo "Installing dependencies from requirements.txt..."
        # Install core scientific packages via conda (faster, better optimized)
        conda install -y numpy scipy numba pandas pyyaml matplotlib -c conda-forge 2>/dev/null || true

        # Install remaining from requirements.txt
        if [[ -f requirements.txt ]]; then
            pip install --quiet -r requirements.txt 2>/dev/null || pip install --quiet numpy scipy numba pandas pyyaml matplotlib seaborn
        fi

        # Create directories
        mkdir -p logs data

        # Verify setup
        echo ""
        echo "Verifying installation..."
        python -c "import numpy; print(f'  numpy {numpy.__version__}')"
        python -c "import scipy; print(f'  scipy {scipy.__version__}')"
        python -c "import numba; print(f'  numba {numba.__version__}')"
        python -c "from numba import njit, prange; print('  numba parallel: OK')"

        echo ""
        echo "Setup complete!"
        conda deactivate
REMOTE_SETUP

    # Upload job scripts
    log_info "Uploading SLURM job scripts..."
    scp -r "$SCRIPT_DIR/jobs" "${CLUSTER_SSH}:~/mcm2026/slurm/"

    log_info "Setup complete! Run './slurm/start.sh run' to submit the pipeline."
}

#
# RUN: Submit the full pipeline with job dependencies
#
cmd_run() {
    log_info "Submitting pipeline to SLURM..."

    # Upload latest scripts and config
    scp -r "$SCRIPT_DIR/jobs" "${CLUSTER_SSH}:~/mcm2026/slurm/"
    scp "$PROJECT_DIR/.env" "${CLUSTER_SSH}:~/mcm2026/.env"

    ssh "$CLUSTER_SSH" 'bash -s' << 'REMOTE_RUN'
        set -e
        cd ~/mcm2026

        # Load config
        if [[ -f .env ]]; then
            set -a
            source .env
            set +a
        fi

        # Make scripts executable
        chmod +x slurm/jobs/*.sh

        mkdir -p logs

        # Submit pipeline with dependencies
        echo "Submitting initialize job..."
        JOB1=$(sbatch --parsable slurm/jobs/initialize.sh)
        echo "  Job ID: $JOB1"

        echo "Submitting backprojection job (waits for $JOB1)..."
        JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/jobs/backproject.sh)
        echo "  Job ID: $JOB2"

        echo "Submitting forward projection job (waits for $JOB2)..."
        JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/jobs/forwardproject.sh)
        echo "  Job ID: $JOB3"

        echo "Submitting export job (waits for $JOB3)..."
        JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/jobs/export.sh)
        echo "  Job ID: $JOB4"

        echo "Submitting zip/transfer job (waits for $JOB4)..."
        JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 slurm/jobs/transfer.sh)
        echo "  Job ID: $JOB5"

        echo ""
        echo "Pipeline submitted!"
        echo "Chain: init($JOB1) -> backproj($JOB2) -> fwdproj($JOB3) -> export($JOB4) -> zip($JOB5)"
        echo ""
        echo "Monitor: squeue -u $USER"
REMOTE_RUN

    log_info "Pipeline submitted! Use './slurm/start.sh status' to monitor."
}

#
# STATUS: Check job status
#
cmd_status() {
    log_info "Job status for $TUFTS_USER:"
    ssh "$CLUSTER_SSH" "squeue -u $TUFTS_USER --format='%.10i %.20j %.8T %.12M %.4D %R' 2>/dev/null || squeue -u \$USER"
}

#
# DOWNLOAD: Download results from cluster
#
cmd_download() {
    log_info "Downloading results from cluster..."

    LOCAL_DIR="${LOCAL_RESULTS_DIR:-./cluster_results}"
    mkdir -p "$LOCAL_DIR"

    # Check for zip file
    LATEST_ZIP=$(ssh "$CLUSTER_SSH" "ls -t ~/mcm2026/results_*.zip 2>/dev/null | head -1" || echo "")

    if [[ -n "$LATEST_ZIP" ]]; then
        log_info "Found: $LATEST_ZIP"
        scp "${CLUSTER_SSH}:${LATEST_ZIP}" "$LOCAL_DIR/"

        ZIPNAME=$(basename "$LATEST_ZIP")
        log_info "Extracting $ZIPNAME..."
        unzip -o "$LOCAL_DIR/$ZIPNAME" -d "$LOCAL_DIR/"
        log_info "Results in $LOCAL_DIR/"
    else
        log_warn "No zip found. Downloading data/ directly..."
        scp -r "${CLUSTER_SSH}:~/mcm2026/data" "$LOCAL_DIR/"
        log_info "Data downloaded to $LOCAL_DIR/data/"
    fi
}

#
# LOGS: View recent logs
#
cmd_logs() {
    log_info "Recent logs:"
    ssh "$CLUSTER_SSH" 'bash -s' << 'REMOTE_LOGS'
        cd ~/mcm2026 2>/dev/null || exit 0
        echo "=== Log files ==="
        ls -lt logs/*.out logs/*.err 2>/dev/null | head -10
        echo ""
        LATEST=$(ls -t logs/*.out 2>/dev/null | head -1)
        if [[ -n "$LATEST" ]]; then
            echo "=== Last 50 lines of $LATEST ==="
            tail -50 "$LATEST"
        fi
REMOTE_LOGS
}

#
# CLEAN: Cancel all jobs
#
cmd_clean() {
    log_warn "Cancelling all SLURM jobs for $TUFTS_USER..."
    ssh "$CLUSTER_SSH" "scancel -u $TUFTS_USER 2>/dev/null || scancel -u \$USER"
    log_info "Jobs cancelled."
}

#
# SSH: Open interactive session
#
cmd_ssh() {
    log_info "Connecting to $CLUSTER_HOST..."
    ssh "$CLUSTER_SSH"
}

#
# HELP
#
cmd_help() {
    cat << 'EOF'
SLURM Pipeline for Region Analysis (Tufts HPC)

Usage: ./slurm/start.sh [command]

Commands:
  setup      Clone repo and setup conda environment on cluster
  run        Submit full pipeline with job dependencies
  status     Check SLURM job status
  download   Download results to local machine
  logs       View recent job logs
  clean      Cancel all your SLURM jobs
  ssh        Open SSH session to cluster
  help       Show this help

Workflow:
  1. cp .env.example .env        # Create config
  2. nano .env                   # Fill in TUFTS_USER, etc.
  3. ./slurm/start.sh setup      # One-time cluster setup
  4. ./slurm/start.sh run        # Submit pipeline
  5. ./slurm/start.sh status     # Monitor jobs
  6. ./slurm/start.sh download   # Get results when done
EOF
}

# Main dispatch
case "${1:-help}" in
    setup)    cmd_setup ;;
    run)      cmd_run ;;
    status)   cmd_status ;;
    download) cmd_download ;;
    logs)     cmd_logs ;;
    clean)    cmd_clean ;;
    ssh)      cmd_ssh ;;
    help|*)   cmd_help ;;
esac
