#!/usr/bin/env python3
"""
Region Analysis for DWTS Vote Distributions.

For each (season, week) event, finds the convex region of valid fan vote
probability vectors that satisfy the premise constraints.

Usage:
    python region-analysis/main.py [--samples N] [--seasons S1,S2,...] [--output PATH]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

log.info("Importing modules...")
from config import get_config
log.info("  config OK")
from events import load_events, Event
log.info("  events OK")
from sampler import sample_valid_votes, SampleResult
log.info("  sampler OK")
from region import compute_region, save_regions, RegionInfo
log.info("  region OK")

DATA_DIR = Path(__file__).parent.parent / "data"


def analyze_event(event: Event, n_samples: int, seed: int) -> RegionInfo:
    """Run full analysis for a single event."""
    log.info("  [MAIN] Starting sampling...")
    sample_result = sample_valid_votes(event, n_samples=n_samples, seed=seed)
    log.info("  [MAIN] Computing convex hull...")
    region_info = compute_region(sample_result)
    log.info("  [MAIN] Done with event.")
    return region_info


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(description="DWTS Vote Region Analysis")
    parser.add_argument("--samples", "-n", type=int, default=cfg.sampling.n_samples,
                        help=f"Number of random samples per event (default: {cfg.sampling.n_samples})")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to analyze")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for results JSON")
    parser.add_argument("--seed", type=int, default=cfg.sampling.seed,
                        help=f"Random seed (default: {cfg.sampling.seed})")
    args = parser.parse_args()

    # Parse seasons
    seasons = None
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]

    # Load events
    log.info("Loading events from main.csv...")
    events = load_events()
    log.info(f"Loaded {len(events)} events")

    if seasons:
        events = [e for e in events if e.season in seasons]
        log.info(f"Filtered to {len(events)} events for seasons {seasons}")

    # Analyze each event
    results: List[RegionInfo] = []
    for i, event in enumerate(events):
        tag = f"S{event.season}W{event.week}"
        if event.is_final:
            tag += " (final)"

        log.info(f"[{i+1}/{len(events)}] Analyzing {tag}: {event.n} contestants...")

        region = analyze_event(event, n_samples=args.samples, seed=args.seed + i)
        results.append(region)

        status = f"valid={region.n_valid}/{region.n_samples} ({region.acceptance_rate:.1%})"
        if region.has_hull:
            status += f", vertices={region.n_vertices}, vol={region.volume:.2e}"
        log.info(f"  {status}")

    # Save results
    out_path = args.output or str(DATA_DIR / "regions.json")
    log.info(f"Saving {len(results)} regions to {out_path}")
    save_regions(results, out_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total events analyzed: {len(results)}")
    print(f"Events with hull: {sum(1 for r in results if r.has_hull)}")
    print(f"Events with 0 valid: {sum(1 for r in results if r.n_valid == 0)}")

    # Acceptance rates by premise
    by_premise = {}
    for r in results:
        if r.premise_type not in by_premise:
            by_premise[r.premise_type] = []
        by_premise[r.premise_type].append(r.acceptance_rate)

    print("\nAcceptance rates by premise:")
    for ptype in sorted(by_premise.keys()):
        rates = by_premise[ptype]
        print(f"  {ptype:<12}: mean={np.mean(rates):.1%}, min={min(rates):.1%}, max={max(rates):.1%}")

    # List events with 0 valid points
    zero_events = [r for r in results if r.n_valid == 0]
    if zero_events:
        print(f"\n*** FAILED EVENTS (0 valid points): {len(zero_events)} ***")
        for r in zero_events:
            final = " [FINAL]" if "FINAL" in r.premise_type else ""
            print(f"  Season {r.season}, Week {r.week}{final} - {r.premise_type}, n={r.n_contestants}")

    # List events with very low acceptance (but > 0)
    low_events = [r for r in results if 0 < r.n_valid < 10]
    if low_events:
        print(f"\n*** LOW SAMPLE EVENTS (<10 valid): {len(low_events)} ***")
        for r in low_events:
            final = " [FINAL]" if "FINAL" in r.premise_type else ""
            print(f"  Season {r.season}, Week {r.week}{final} - {r.n_valid} valid, {r.premise_type}")


if __name__ == "__main__":
    import numpy as np
    main()
