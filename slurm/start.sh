#!/bin/bash
#
# SLURM Pipeline Orchestrator for Region Analysis
#
# Usage:
#   ./slurm/start.sh [command]
#
# Commands:
#   setup      - Clone repo and setup environment on cluster
#   run        - Submit full pipeline as chained jobs (init -> back -> fwd -> final -> export -> verify -> zip)
#   pipeline   - Submit full pipeline as single job (uses main.py)
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

CLUSTER_HOST="login.pax.tufts.edu"
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
    log_info "Setting up environment on Tufts HPC (single connection)..."

    # Do everything in one SSH session using ControlMaster
    ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-%r@%h:%p -o ControlPersist=60 "$CLUSTER_SSH" "mkdir -p ~/mcm2026/slurm/jobs"

    # Upload all files in one scp call
    scp -o ControlPath=/tmp/ssh-%r@%h:%p \
        "$PROJECT_DIR/.env" \
        "$PROJECT_DIR/environment.yml" \
        "$SCRIPT_DIR/jobs/"*.sh \
        "${CLUSTER_SSH}:~/mcm2026/"

    # Move job scripts to correct location and run setup
    ssh -o ControlPath=/tmp/ssh-%r@%h:%p "$CLUSTER_SSH" 'bash -s' << 'REMOTE_SETUP'
        set -e
        cd ~/mcm2026

        # Move job scripts to correct location
        mv -f *.sh slurm/jobs/ 2>/dev/null || true

        # Load env vars
        if [[ -f .env ]]; then
            set -a
            source .env
            set +a
        fi

        # Clone or pull repo
        REPO_URL="${GITHUB_REPO:-https://github.com/your-username/mcm2026.git}"
        if [[ -d .git ]]; then
            echo "Pulling latest from GitHub..."
            git pull || echo "Git pull failed, continuing"
        else
            echo "Cloning repository..."
            cd ~
            rm -rf mcm2026
            git clone "$REPO_URL" mcm2026
            cd mcm2026
        fi

        mkdir -p logs data slurm/jobs
        chmod +x slurm/jobs/*.sh

        # Load miniforge
        module purge
        module load miniforge/24.11.2-py312 2>/dev/null || \
        module load miniforge/24.7.1-py312 2>/dev/null || \
        module load miniforge 2>/dev/null

        echo "Conda: $(which conda)"

        # Create/update conda environment
        if [[ ! -d mcm_env ]]; then
            echo "Creating conda environment..."
            conda env create -f environment.yml -p ~/mcm2026/mcm_env
        else
            echo "Updating conda environment..."
            conda env update -f environment.yml -p ~/mcm2026/mcm_env --prune || true
        fi

        source activate ~/mcm2026/mcm_env

        echo ""
        echo "Verifying installation..."
        python -c "import numpy; print(f'  numpy {numpy.__version__}')"
        python -c "import scipy; print(f'  scipy {scipy.__version__}')"
        python -c "import numba; print(f'  numba {numba.__version__}')"
        python -c "from numba import njit, prange; print('  numba parallel: OK')"

        echo ""
        echo "Setup complete!"
        source deactivate || true
REMOTE_SETUP

    log_info "Setup complete! Run './slurm/start.sh run' or './slurm/start.sh pipeline'"
}

#
# RUN: Submit chained jobs (each step as separate job with dependencies)
#
cmd_run() {
    log_info "Submitting chained pipeline to SLURM..."

    # Check local git status
    LOCAL_COMMIT=$(git rev-parse HEAD)
    log_info "Local commit: $LOCAL_COMMIT"

    # Single connection for all operations
    ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-%r@%h:%p -o ControlPersist=60 "$CLUSTER_SSH" "mkdir -p ~/mcm2026/slurm/jobs ~/mcm2026/logs ~/mcm2026/data"
    scp -o ControlPath=/tmp/ssh-%r@%h:%p "$SCRIPT_DIR/jobs/"*.sh "${CLUSTER_SSH}:~/mcm2026/slurm/jobs/"
    scp -o ControlPath=/tmp/ssh-%r@%h:%p "$PROJECT_DIR/region-analysis/config.yaml" "${CLUSTER_SSH}:~/mcm2026/region-analysis/config.yaml"

    ssh -o ControlPath=/tmp/ssh-%r@%h:%p "$CLUSTER_SSH" 'bash -s' << 'REMOTE_RUN'
        set -e
        cd ~/mcm2026

        echo "Pulling latest from GitHub..."
        git fetch origin
        git reset --hard origin/main || git reset --hard origin/master

        REMOTE_COMMIT=$(git rev-parse HEAD)
        echo ""
        echo "=========================================="
        echo "REMOTE COMMIT: $REMOTE_COMMIT"
        echo "=========================================="
        echo ""

        echo "Cleaning old data files..."
        rm -f data/regions*.json data/regions*.csv data/regions*.txt data/events.json
        rm -f logs/*.out logs/*.err
        rm -rf __pycache__ region-analysis/__pycache__

        chmod +x slurm/jobs/*.sh
        mkdir -p logs data

        echo "Pipeline: init -> backproj -> fwdproj -> finalize -> export -> verify -> zip"
        echo ""

        echo "1. Submitting initialize..."
        JOB1=$(sbatch --parsable slurm/jobs/initialize.sh)
        echo "   Job ID: $JOB1"

        echo "2. Submitting backprojection (waits for $JOB1)..."
        JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/jobs/backproject.sh)
        echo "   Job ID: $JOB2"

        echo "3. Submitting forward projection (waits for $JOB2)..."
        JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/jobs/forwardproject.sh)
        echo "   Job ID: $JOB3"

        echo "4. Submitting finalize (waits for $JOB3)..."
        JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/jobs/finalize.sh)
        echo "   Job ID: $JOB4"

        echo "5. Submitting export (waits for $JOB4)..."
        JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 slurm/jobs/export.sh)
        echo "   Job ID: $JOB5"

        echo "6. Submitting verify (waits for $JOB5)..."
        JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 slurm/jobs/verify.sh)
        echo "   Job ID: $JOB6"

        echo "7. Submitting zip/transfer (waits for $JOB6)..."
        JOB7=$(sbatch --parsable --dependency=afterok:$JOB6 slurm/jobs/transfer.sh)
        echo "   Job ID: $JOB7"

        echo ""
        echo "Pipeline submitted!"
        echo "Chain: init($JOB1) -> back($JOB2) -> fwd($JOB3) -> final($JOB4) -> export($JOB5) -> verify($JOB6) -> zip($JOB7)"
        echo ""
        echo "Monitor: squeue -u $USER"
REMOTE_RUN

    log_info "Pipeline submitted! Use './slurm/start.sh status' to monitor."
}

#
# PIPELINE: Submit full pipeline as single job (uses main.py)
#
cmd_pipeline() {
    log_info "Submitting full pipeline as single job..."

    # Check local git status
    LOCAL_COMMIT=$(git rev-parse HEAD)
    log_info "Local commit: $LOCAL_COMMIT"

    # Single connection for all operations
    ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-%r@%h:%p -o ControlPersist=60 "$CLUSTER_SSH" "mkdir -p ~/mcm2026/slurm/jobs ~/mcm2026/logs ~/mcm2026/data"
    scp -o ControlPath=/tmp/ssh-%r@%h:%p "$SCRIPT_DIR/jobs/"*.sh "${CLUSTER_SSH}:~/mcm2026/slurm/jobs/"
    scp -o ControlPath=/tmp/ssh-%r@%h:%p "$PROJECT_DIR/region-analysis/config.yaml" "${CLUSTER_SSH}:~/mcm2026/region-analysis/config.yaml"

    ssh -o ControlPath=/tmp/ssh-%r@%h:%p "$CLUSTER_SSH" 'bash -s' << 'REMOTE_PIPELINE'
        set -e
        cd ~/mcm2026

        echo "Pulling latest from GitHub..."
        git fetch origin
        git reset --hard origin/main || git reset --hard origin/master

        REMOTE_COMMIT=$(git rev-parse HEAD)
        echo ""
        echo "=========================================="
        echo "REMOTE COMMIT: $REMOTE_COMMIT"
        echo "=========================================="
        echo ""

        echo "Cleaning old data files..."
        rm -f data/regions*.json data/regions*.csv data/regions*.txt data/events.json
        rm -f logs/*.out logs/*.err
        rm -rf __pycache__ region-analysis/__pycache__

        chmod +x slurm/jobs/*.sh
        mkdir -p logs data

        echo "Submitting full pipeline job (main.py)..."
        JOB1=$(sbatch --parsable slurm/jobs/pipeline.sh)
        echo "  Job ID: $JOB1"

        echo "Submitting zip/transfer (waits for $JOB1)..."
        JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/jobs/transfer.sh)
        echo "  Job ID: $JOB2"

        echo ""
        echo "Pipeline submitted: pipeline($JOB1) -> zip($JOB2)"
        echo "Monitor: squeue -u $USER"
REMOTE_PIPELINE

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

    LATEST_ZIP=$(ssh "$CLUSTER_SSH" "ls -t ~/mcm2026/results_*.zip 2>/dev/null | head -1" || echo "")

    if [[ -n "$LATEST_ZIP" ]]; then
        log_info "Found: $LATEST_ZIP"
        scp "${CLUSTER_SSH}:${LATEST_ZIP}" "$LOCAL_DIR/"

        ZIPNAME=$(basename "$LATEST_ZIP")
        # Extract timestamp from filename (results_YYYYMMDD_HHMMSS.zip -> run_YYYYMMDD_HHMMSS)
        TIMESTAMP=$(echo "$ZIPNAME" | sed 's/results_\(.*\)\.zip/\1/')
        RUN_DIR="$LOCAL_DIR/run_$TIMESTAMP"

        log_info "Extracting to $RUN_DIR..."
        mkdir -p "$RUN_DIR"
        unzip -o "$LOCAL_DIR/$ZIPNAME" -d "$RUN_DIR/"
        log_info "Results in $RUN_DIR/"
    else
        log_warn "No zip found. Downloading data/ directly..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        RUN_DIR="$LOCAL_DIR/run_$TIMESTAMP"
        mkdir -p "$RUN_DIR"
        scp -r "${CLUSTER_SSH}:~/mcm2026/data" "$RUN_DIR/"
        log_info "Data downloaded to $RUN_DIR/"
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
  run        Submit chained jobs: init -> back -> fwd -> final -> export -> verify -> zip
  pipeline   Submit as single job using main.py (simpler, but no partial recovery)
  status     Check SLURM job status
  download   Download results to local machine
  logs       View recent job logs
  clean      Cancel all your SLURM jobs
  ssh        Open SSH session to cluster
  help       Show this help

Workflow:
  1. cp .env.example .env        # Create config
  2. nano .env                   # Fill in TUFTS_USER, GITHUB_REPO, etc.
  3. ./slurm/start.sh setup      # One-time cluster setup
  4. ./slurm/start.sh run        # Submit pipeline (or: pipeline)
  5. ./slurm/start.sh status     # Monitor jobs
  6. ./slurm/start.sh download   # Get results when done

Pipeline steps:
  1. Initialize   - Sample valid vote distributions, compute convex hulls
  2. Backproject  - Constrain earlier weeks by later weeks
  3. Forward      - Constrain later weeks by earlier weeks
  4. Finalize     - Sample from hull, verify points, create point clouds
  5. Export       - Export to CSV and TXT summaries
  6. Verify       - Verify results against elimination constraints
  7. Zip          - Package results for download

Note: Experiment settings (samples, seed, seasons) are in region-analysis/config.yaml
EOF
}

# Main dispatch
case "${1:-help}" in
    setup)    cmd_setup ;;
    run)      cmd_run ;;
    pipeline) cmd_pipeline ;;
    status)   cmd_status ;;
    download) cmd_download ;;
    logs)     cmd_logs ;;
    clean)    cmd_clean ;;
    ssh)      cmd_ssh ;;
    help|*)   cmd_help ;;
esac
