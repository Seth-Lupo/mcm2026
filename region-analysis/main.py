#!/usr/bin/env python3
"""
DWTS Vote Region Analysis - Full Pipeline

Runs the complete analysis pipeline:
  1. Initialize   - Sample valid vote distributions, compute convex hulls
  2. Backproject  - Constrain earlier weeks by later weeks
  3. Forward      - Constrain later weeks by earlier weeks
  4. Finalize     - Sample from hull, verify points, create point clouds
  5. Export       - Export to CSV and TXT summaries
  6. Verify       - Verify results against elimination constraints

Usage:
    python region-analysis/main.py [options]
"""
import sys
from pathlib import Path

# Add script dir to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import subprocess
import argparse
import time
from datetime import datetime

from config import get_config

DATA_DIR = SCRIPT_DIR.parent / "data"


def run_step(name: str, script: str, args: list) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"{'='*70}")

    cmd = [sys.executable, str(SCRIPT_DIR / script)] + args
    print(f"  Command: {' '.join(cmd)}\n")

    start = time.time()

    # Stream output directly to terminal
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(
        description="DWTS Vote Region Analysis - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sampling options (defaults from config)
    parser.add_argument("--samples", "-n", type=int, default=cfg.sampling.n_samples,
                        help=f"Number of random samples per event (default: {cfg.sampling.n_samples})")
    parser.add_argument("--season", type=int, default=None,
                        help="Single season to analyze (e.g., --season 32)")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to analyze (default: all)")
    parser.add_argument("--seed", type=int, default=cfg.sampling.seed,
                        help=f"Random seed for reproducibility (default: {cfg.sampling.seed})")

    # Hull options (defaults from config)
    parser.add_argument("--hull-validity-samples", type=int, default=cfg.hull.volume_samples,
                        help=f"Monte Carlo samples for hull validity (default: {cfg.hull.volume_samples})")

    # Finalization options
    parser.add_argument("--hull-samples", type=int, default=10000,
                        help="Samples from hull for finalization (default: 10000)")
    parser.add_argument("--simplex-samples", type=int, default=cfg.hull.volume_samples,
                        help=f"Simplex samples for accuracy estimation (default: {cfg.hull.volume_samples})")

    # Skip options
    parser.add_argument("--skip-init", action="store_true",
                        help="Skip initialization step")
    parser.add_argument("--skip-backproject", action="store_true",
                        help="Skip backprojection step")
    parser.add_argument("--skip-forward", action="store_true",
                        help="Skip forward projection step")
    parser.add_argument("--skip-finalize", action="store_true",
                        help="Skip finalization step")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export step")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification step")

    # Processing options
    parser.add_argument("--sequential", action="store_true",
                        help="Disable parallel processing")

    args = parser.parse_args()

    # Track timing
    pipeline_start = time.time()
    steps_run = 0
    steps_failed = 0

    # Handle --season (single) vs --seasons (multiple)
    seasons_str = None
    if args.season:
        seasons_str = str(args.season)
    elif args.seasons:
        seasons_str = args.seasons

    print("="*70)
    print("  DWTS VOTE REGION ANALYSIS PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\n  Config: samples={args.samples}, seed={args.seed}")
    print(f"  Hull validity samples: {args.hull_validity_samples}")
    print(f"  Finalize: hull_samples={args.hull_samples}, simplex_samples={args.simplex_samples}")
    if seasons_str:
        print(f"  Seasons: {seasons_str}")
    else:
        print(f"  Seasons: all")

    # Build common args
    common_args = []
    if seasons_str:
        common_args.extend(["--seasons", seasons_str])
    if args.sequential:
        common_args.append("--sequential")

    # Step 1: Initialize
    if not args.skip_init:
        init_args = common_args.copy()
        init_args.extend(["--samples", str(args.samples)])
        init_args.extend(["--seed", str(args.seed)])
        init_args.extend(["--hull-validity-samples", str(args.hull_validity_samples)])
        init_args.extend(["--output", str(DATA_DIR / "regions.json")])

        if run_step("Initialize", "initialize.py", init_args):
            steps_run += 1
        else:
            steps_failed += 1
            print("\n*** Initialization failed, stopping pipeline ***")
            return 1
    else:
        print("\n  [SKIP] Initialize")

    # Step 2: Backproject
    if not args.skip_backproject:
        back_args = common_args.copy()
        back_args.extend(["--input", str(DATA_DIR / "regions.json")])
        back_args.extend(["--output", str(DATA_DIR / "regions-backprojected.json")])

        if run_step("Backproject", "backprojection.py", back_args):
            steps_run += 1
        else:
            steps_failed += 1
            print("\n*** Backprojection failed, continuing... ***")
    else:
        print("\n  [SKIP] Backproject")

    # Step 3: Forward project
    if not args.skip_forward:
        fwd_args = common_args.copy()
        fwd_args.extend(["--input", str(DATA_DIR / "regions-backprojected.json")])
        fwd_args.extend(["--output", str(DATA_DIR / "regions-forwardprojected.json")])

        if run_step("Forward Project", "forwardprojection.py", fwd_args):
            steps_run += 1
        else:
            steps_failed += 1
            print("\n*** Forward projection failed, continuing... ***")
    else:
        print("\n  [SKIP] Forward Project")

    # Step 4: Finalize
    if not args.skip_finalize:
        final_args = common_args.copy()
        final_args.extend(["--input", str(DATA_DIR / "regions-forwardprojected.json")])
        final_args.extend(["--output", str(DATA_DIR / "regions-finalized.json")])
        final_args.extend(["--hull-samples", str(args.hull_samples)])
        final_args.extend(["--simplex-samples", str(args.simplex_samples)])
        final_args.extend(["--seed", str(args.seed)])

        if run_step("Finalize", "finalize.py", final_args):
            steps_run += 1
        else:
            steps_failed += 1
            print("\n*** Finalization failed, continuing... ***")
    else:
        print("\n  [SKIP] Finalize")

    # Step 5: Export
    if not args.skip_export:
        export_args = ["--data-dir", str(DATA_DIR)]

        if run_step("Export", "export.py", export_args):
            steps_run += 1
        else:
            steps_failed += 1
            print("\n*** Export failed, continuing... ***")
    else:
        print("\n  [SKIP] Export")

    # Step 6: Verify
    if not args.skip_verify:
        verify_args = common_args.copy()
        verify_args.extend(["--regions", str(DATA_DIR / "regions-finalized.json")])
        verify_args.extend(["--events", str(DATA_DIR / "events.json")])

        if run_step("Verify", "verify.py", verify_args):
            steps_run += 1
        else:
            steps_failed += 1
            print("\n*** Verification failed ***")
    else:
        print("\n  [SKIP] Verify")

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print(f"  Steps run: {steps_run}")
    print(f"  Steps failed: {steps_failed}")
    print(f"  Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # List output files
    print("\n  Output files:")
    output_files = [
        "regions.json",
        "regions-backprojected.json",
        "regions-forwardprojected.json",
        "regions-finalized.json",
        "votes.csv",
        "regions.csv",
        "regions-backprojected.csv",
        "regions-forwardprojected.csv",
        "regions-finalized.csv",
        "regions.txt",
        "regions-backprojected.txt",
        "regions-forwardprojected.txt",
        "regions-finalized.txt",
    ]
    for fname in output_files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"    {fname}: {size_str}")

    print("="*70)

    return 1 if steps_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
