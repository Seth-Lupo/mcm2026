#!/usr/bin/env python3
"""
Region Analysis for DWTS Vote Distributions.

For each (season, week) event, finds the convex region of valid fan vote
probability vectors that satisfy the premise constraints.

Usage:
    python region-analysis/initialize.py [--samples N] [--seasons S1,S2,...] [--output PATH] [--hull-validity-samples N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

log.info("Importing modules...")
from config import get_config
log.info("  config OK")
from events import load_events, save_events, Event
log.info("  events OK")
from sampler import sample_valid_votes, SampleResult
log.info("  sampler OK")
from region import compute_region, save_regions, RegionInfo
log.info("  region OK")

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Worker function for parallel processing
# =============================================================================

def _analyze_event_worker(args: Tuple[Dict[str, Any], int, int, int]) -> Dict[str, Any]:
    """
    Worker function to analyze a single event.
    Takes serialized event data and returns serialized result with logs.
    """
    event_dict, n_samples, seed, hull_validity_samples = args

    # Reconstruct Event object (n is a property, not a constructor arg)
    from events import Event
    event = Event(
        season=event_dict["season"],
        week=event_dict["week"],
        contestants=event_dict["contestants"],
        judge_scores=np.array(event_dict["judge_scores"]),
        eliminated=set(event_dict["eliminated"]),
        is_final=event_dict["is_final"],
        placements=event_dict.get("placements"),
    )

    # Capture logs
    log_lines = []

    # Run analysis
    log_lines.append(f"  [SAMPLE] Starting sampling...")
    sample_result = sample_valid_votes(event, n_samples=n_samples, seed=seed)
    log_lines.append(f"  [SAMPLE] {sample_result.n_valid}/{n_samples} valid ({sample_result.acceptance_rate:.1%})")

    log_lines.append(f"  [HULL] Computing convex hull...")
    region_info = compute_region(sample_result, hull_validity_samples=hull_validity_samples)

    if region_info.has_hull:
        log_lines.append(f"  [HULL] {region_info.n_vertices} vertices, vol={region_info.volume:.2e}, hull_valid={region_info.hull_validity:.1%}")
    else:
        log_lines.append(f"  [HULL] No hull (insufficient valid points)")

    return {
        "region": region_info.to_dict(),
        "logs": log_lines,
        "season": event.season,
        "week": event.week,
        "is_final": event.is_final,
    }


def analyze_event(event: Event, n_samples: int, seed: int, hull_validity_samples: int = 10000) -> RegionInfo:
    """Run full analysis for a single event (sequential version)."""
    log.info("  [MAIN] Starting sampling...")
    sample_result = sample_valid_votes(event, n_samples=n_samples, seed=seed)
    log.info("  [MAIN] Computing convex hull...")
    region_info = compute_region(sample_result, hull_validity_samples=hull_validity_samples)
    log.info("  [MAIN] Done with event.")
    return region_info


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(description="DWTS Vote Region Analysis")
    parser.add_argument("--samples", "-n", type=int, default=cfg.sampling.n_samples,
                        help=f"Number of random samples per event (default: {cfg.sampling.n_samples})")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to analyze")
    parser.add_argument("--week", "-w", default=None,
                        help="Week to analyze (number or 'final')")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for results JSON")
    parser.add_argument("--seed", type=int, default=cfg.sampling.seed,
                        help=f"Random seed (default: {cfg.sampling.seed})")
    parser.add_argument("--sequential", action="store_true",
                        help="Process events sequentially (disable parallelization)")
    parser.add_argument("--hull-validity-samples", type=int, default=10000,
                        help="Number of Monte Carlo samples for hull validity estimation (default: 10000)")
    args = parser.parse_args()

    # Parse seasons
    seasons = None
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(",")]

    # Load events
    log.info("Loading events from main.csv...")
    events = load_events()
    log.info(f"Loaded {len(events)} events")

    # Save preprocessed events
    save_events(events)
    log.info(f"Saved events to data/events.json")

    if seasons:
        events = [e for e in events if e.season in seasons]
        log.info(f"Filtered to {len(events)} events for seasons {seasons}")

    if args.week:
        if args.week.lower() == "final":
            events = [e for e in events if e.is_final]
            log.info(f"Filtered to {len(events)} final events")
        else:
            week_num = int(args.week)
            events = [e for e in events if e.week == week_num and not e.is_final]
            log.info(f"Filtered to {len(events)} events for week {week_num}")

    n_events = len(events)
    n_workers = min(cfg.parallel.n_workers, n_events)

    if args.sequential or n_events <= 1 or n_workers <= 1:
        # Sequential processing
        results: List[RegionInfo] = []
        for i, event in enumerate(events):
            tag = f"S{event.season}W{event.week}"
            if event.is_final:
                tag += " (final)"

            log.info(f"[{i+1}/{len(events)}] Analyzing {tag}: {event.n} contestants...")

            region = analyze_event(event, n_samples=args.samples, seed=args.seed + i, hull_validity_samples=args.hull_validity_samples)
            results.append(region)

            status = f"valid={region.n_valid}/{region.n_samples} ({region.acceptance_rate:.1%})"
            if region.has_hull:
                status += f", vertices={region.n_vertices}, vol={region.volume:.2e}, hull_valid={region.hull_validity:.1%}"
            log.info(f"  {status}")
    else:
        # Parallel processing
        log.info(f"Processing {n_events} events using {n_workers} workers...")

        # Serialize events for workers
        event_tasks = []
        for i, event in enumerate(events):
            event_dict = {
                "season": event.season,
                "week": event.week,
                "contestants": event.contestants,
                "judge_scores": event.judge_scores.tolist(),
                "eliminated": list(event.eliminated),
                "is_final": event.is_final,
                "placements": event.placements,
            }
            event_tasks.append((event_dict, args.samples, args.seed + i, args.hull_validity_samples))

        # Process in parallel
        results_list = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_analyze_event_worker, task): i
                for i, task in enumerate(event_tasks)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results_list.append((idx, result))
                completed += 1

                # Emit logs for this event
                tag = f"S{result['season']}W{result['week']}"
                if result['is_final']:
                    tag += " (final)"
                log.info(f"[{completed}/{n_events}] Completed {tag}")
                for line in result['logs']:
                    log.info(line)

        # Sort by original order and extract RegionInfo
        results_list.sort(key=lambda x: x[0])

        # Convert back to RegionInfo objects
        from region import RegionInfo
        results = []
        for _, result_dict in results_list:
            region_data = result_dict["region"]
            event_data = region_data["event"]
            sampling_data = region_data["sampling"]
            region_info_data = region_data["region"]

            region = RegionInfo(
                season=event_data["season"],
                week=event_data["week"],
                is_final=event_data["is_final"],
                premise_type=event_data["premise_type"],
                contestants=event_data["contestants"],
                n_contestants=event_data["n_contestants"],
                n_samples=sampling_data["n_samples"],
                n_valid=sampling_data["n_valid"],
                acceptance_rate=sampling_data["acceptance_rate"],
                has_hull=region_info_data["has_hull"],
                n_vertices=region_info_data["n_vertices"],
                volume=region_info_data["volume"],
                relative_volume=region_info_data["relative_volume"],
                hull_validity=region_info_data.get("hull_validity", 0.0),
                hull_validity_samples=region_info_data.get("hull_validity_samples", 0),
                diameter=region_info_data["diameter"],
                centroid=np.array(region_info_data["centroid"]) if region_info_data.get("centroid") else None,
                dim_bounds=np.array([[b["min"], b["max"]] for b in region_info_data["dim_bounds"]]) if region_info_data.get("dim_bounds") else None,
                vertices=np.array(region_info_data["vertices"]) if region_info_data.get("vertices") else None,
            )
            results.append(region)

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

    # Hull validity by premise (how much of the convex hull satisfies the premise)
    hull_validity_by_premise = {}
    for r in results:
        if r.has_hull and r.hull_validity_samples > 0:
            if r.premise_type not in hull_validity_by_premise:
                hull_validity_by_premise[r.premise_type] = []
            hull_validity_by_premise[r.premise_type].append(r.hull_validity)

    if hull_validity_by_premise:
        print("\nHull validity by premise (fraction of hull satisfying premise):")
        for ptype in sorted(hull_validity_by_premise.keys()):
            validities = hull_validity_by_premise[ptype]
            print(f"  {ptype:<12}: mean={np.mean(validities):.1%}, min={min(validities):.1%}, max={max(validities):.1%}")

        # Identify potentially non-convex regions (hull validity < 100%)
        low_validity = [(r, r.hull_validity) for r in results if r.has_hull and r.hull_validity < 0.99]
        if low_validity:
            print(f"\n*** POTENTIALLY NON-CONVEX REGIONS (hull validity < 99%): {len(low_validity)} ***")
            low_validity.sort(key=lambda x: x[1])
            for r, validity in low_validity[:10]:  # Show worst 10
                final = " [FINAL]" if r.is_final else ""
                print(f"  Season {r.season}, Week {r.week}{final} - {r.premise_type}: hull_valid={validity:.1%}")

    # List events with 0 valid points
    zero_events = [r for r in results if r.n_valid == 0]
    if zero_events:
        print(f"\n*** FAILED EVENTS (0 valid points): {len(zero_events)} ***")
        for r in zero_events:
            final = " [FINAL]" if r.is_final else ""
            print(f"  Season {r.season}, Week {r.week}{final} - {r.premise_type}, n={r.n_contestants}")

    # List events with very low acceptance (but > 0)
    low_events = [r for r in results if 0 < r.n_valid < 10]
    if low_events:
        print(f"\n*** LOW SAMPLE EVENTS (<10 valid): {len(low_events)} ***")
        for r in low_events:
            final = " [FINAL]" if r.is_final else ""
            print(f"  Season {r.season}, Week {r.week}{final} - {r.n_valid} valid, {r.premise_type}")


if __name__ == "__main__":
    main()
