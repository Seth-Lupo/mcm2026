#!/usr/bin/env python3
"""
Verification: Check that region centroids/vertices satisfy elimination constraints.

This module verifies that the computed regions (centroids, vertices, bounds) are
consistent with the actual DWTS elimination outcomes using the premise logic.

Fully parallelized using config.parallel.n_workers CPUs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import get_config

# Load simple-models/premises.py directly for full access
import importlib.util
_premises_path = Path(__file__).parent.parent / "simple-models" / "premises.py"
_spec = importlib.util.spec_from_file_location("simple_premises", _premises_path)
_premises = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_premises)

PremiseType = _premises.PremiseType
get_premise_type = _premises.get_premise_type
validate = _premises.validate
predict = _premises.predict
combine_by_rank = _premises.combine_by_rank
combine_by_percent = _premises.combine_by_percent
scores_to_ranks = _premises.scores_to_ranks
scores_to_percent = _premises.scores_to_percent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class Event:
    """Parsed event data from events.json."""
    season: int
    week: int
    is_final: bool
    contestants: List[str]
    judge_scores: np.ndarray
    eliminated: Set[str]
    placements: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            season=d["season"],
            week=d["week"],
            is_final=d.get("is_final", False),
            contestants=d["contestants"],
            judge_scores=np.array(d["judge_scores"], dtype=np.float64),
            eliminated=set(d.get("eliminated", [])),
            placements=np.array(d["placements"]) if "placements" in d else None,
        )

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to serializable dict for multiprocessing."""
        return {
            "season": self.season,
            "week": self.week,
            "is_final": self.is_final,
            "contestants": self.contestants,
            "judge_scores": self.judge_scores.tolist(),
            "eliminated": list(self.eliminated),
            "placements": self.placements.tolist() if self.placements is not None else None,
        }


@dataclass
class Region:
    """Parsed region data from regions JSON."""
    season: int
    week: int
    is_final: bool
    contestants: List[str]
    centroid: Optional[np.ndarray]
    vertices: Optional[np.ndarray]
    dim_bounds: Optional[np.ndarray]
    representative_point: Optional[np.ndarray] = None  # nearest-to-mean from finalization

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Region":
        event = d.get("event", d)
        region = d.get("region", {})

        centroid = None
        if "centroid" in region and region["centroid"]:
            centroid = np.array(region["centroid"], dtype=np.float64)

        vertices = None
        if "vertices" in region and region["vertices"]:
            vertices = np.array(region["vertices"], dtype=np.float64)

        dim_bounds = None
        if "dim_bounds" in region and region["dim_bounds"]:
            bounds_list = region["dim_bounds"]
            dim_bounds = np.array([[b["min"], b["max"]] for b in bounds_list], dtype=np.float64)

        representative_point = None
        if "representative_point" in region and region["representative_point"]:
            representative_point = np.array(region["representative_point"], dtype=np.float64)

        return cls(
            season=event["season"],
            week=event["week"],
            is_final=event.get("is_final", False),
            contestants=event["contestants"],
            centroid=centroid,
            vertices=vertices,
            dim_bounds=dim_bounds,
            representative_point=representative_point,
        )

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to serializable dict for multiprocessing."""
        return {
            "season": self.season,
            "week": self.week,
            "is_final": self.is_final,
            "contestants": self.contestants,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "vertices": self.vertices.tolist() if self.vertices is not None else None,
            "dim_bounds": self.dim_bounds.tolist() if self.dim_bounds is not None else None,
            "representative_point": self.representative_point.tolist() if self.representative_point is not None else None,
        }

    @classmethod
    def from_serializable(cls, d: Dict[str, Any]) -> "Region":
        """Reconstruct from serializable dict."""
        return cls(
            season=d["season"],
            week=d["week"],
            is_final=d["is_final"],
            contestants=d["contestants"],
            centroid=np.array(d["centroid"], dtype=np.float64) if d["centroid"] else None,
            vertices=np.array(d["vertices"], dtype=np.float64) if d["vertices"] else None,
            dim_bounds=np.array(d["dim_bounds"], dtype=np.float64) if d["dim_bounds"] else None,
            representative_point=np.array(d["representative_point"], dtype=np.float64) if d.get("representative_point") else None,
        )


def load_events(path: Path) -> Dict[Tuple[int, int], Event]:
    """Load events.json and index by (season, week)."""
    with open(path) as f:
        data = json.load(f)

    events = {}
    for d in data:
        event = Event.from_dict(d)
        events[(event.season, event.week)] = event

    return events


def load_regions(path: Path) -> List[Region]:
    """Load regions JSON file."""
    with open(path) as f:
        data = json.load(f)

    return [Region.from_dict(d) for d in data]


# =============================================================================
# Verification Logic (Single Point)
# =============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a single point."""
    valid: bool
    premise_type: str  # Use string for serialization
    predicted_eliminated: List[str]
    actual_eliminated: List[str]
    margin: float


def verify_point_core(
    fan_votes: np.ndarray,
    contestants: List[str],
    judge_scores: np.ndarray,
    eliminated: Set[str],
    placements: Optional[np.ndarray],
    season: int,
    is_final: bool,
) -> VerificationResult:
    """
    Core verification logic for a single point.
    Designed for use in parallel workers.
    """
    n_elim = len(eliminated)
    premise_type = get_premise_type(season, n_elim, is_final)

    # Get prediction
    outcome = predict(
        premise_type,
        contestants,
        judge_scores,
        fan_votes,
        n_elim=max(1, n_elim),
    )

    # Calculate margin
    combined = outcome.combined_scores
    if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI, PremiseType.RR_B2, PremiseType.RR_B2_MULTI):
        sorted_combined = np.sort(combined)[::-1]
        if n_elim > 0 and n_elim < len(combined):
            margin = sorted_combined[n_elim - 1] - sorted_combined[n_elim]
        else:
            margin = 0.0
    else:
        sorted_combined = np.sort(combined)
        if n_elim > 0 and n_elim < len(combined):
            margin = sorted_combined[n_elim] - sorted_combined[n_elim - 1]
        else:
            margin = 0.0

    # Validate
    if is_final and placements is not None:
        is_valid = validate(
            premise_type, contestants, judge_scores, fan_votes,
            actual_placements=placements,
        )
    else:
        is_valid = validate(
            premise_type, contestants, judge_scores, fan_votes,
            actual_eliminated=eliminated,
        )

    return VerificationResult(
        valid=is_valid,
        premise_type=premise_type.name,
        predicted_eliminated=list(outcome.eliminated),
        actual_eliminated=list(eliminated),
        margin=float(margin),
    )


# =============================================================================
# Parallel Worker Functions (Module-Level for Pickling)
# =============================================================================

# Global shared data for workers
_worker_events: Dict[Tuple[int, int], Dict] = {}


def _init_worker(events_serialized: Dict[Tuple[int, int], Dict]):
    """Initialize worker with shared event data."""
    global _worker_events
    _worker_events = events_serialized


def _verify_vertex_batch(args: Tuple[List[List[float]], int, int]) -> List[bool]:
    """
    Verify a batch of vertices for a specific region.
    Returns list of booleans indicating validity.
    """
    vertices_batch, season, week = args
    global _worker_events

    event_data = _worker_events.get((season, week))
    if event_data is None:
        return [False] * len(vertices_batch)

    contestants = event_data["contestants"]
    judge_scores = np.array(event_data["judge_scores"], dtype=np.float64)
    eliminated = set(event_data["eliminated"])
    placements = np.array(event_data["placements"]) if event_data["placements"] else None
    is_final = event_data["is_final"]

    results = []
    for vertex in vertices_batch:
        fan_votes = np.array(vertex, dtype=np.float64)
        result = verify_point_core(
            fan_votes, contestants, judge_scores, eliminated, placements, season, is_final
        )
        results.append(result.valid)

    return results


def _verify_region_task(args: Tuple[Dict, int, int]) -> Dict[str, Any]:
    """
    Verify a single region (centroid + vertices).
    Returns serializable result dict.
    """
    region_data, max_vertices, batch_size = args
    global _worker_events

    season = region_data["season"]
    week = region_data["week"]
    is_final = region_data["is_final"]

    event_data = _worker_events.get((season, week))
    if event_data is None:
        return {
            "season": season,
            "week": week,
            "error": "No event found",
        }

    contestants = event_data["contestants"]
    judge_scores = np.array(event_data["judge_scores"], dtype=np.float64)
    eliminated = set(event_data["eliminated"])
    placements = np.array(event_data["placements"]) if event_data["placements"] else None

    # Verify centroid
    centroid_valid = None
    centroid_margin = None
    centroid_predicted = None

    if region_data["centroid"] is not None:
        centroid = np.array(region_data["centroid"], dtype=np.float64)
        result = verify_point_core(
            centroid, contestants, judge_scores, eliminated, placements, season, is_final
        )
        centroid_valid = result.valid
        centroid_margin = result.margin
        centroid_predicted = result.predicted_eliminated

    # Verify representative point (nearest-to-mean from finalization)
    representative_valid = None
    if region_data.get("representative_point") is not None:
        representative = np.array(region_data["representative_point"], dtype=np.float64)
        result = verify_point_core(
            representative, contestants, judge_scores, eliminated, placements, season, is_final
        )
        representative_valid = result.valid

    # Verify vertices
    n_vertices = 0
    n_vertices_valid = 0
    invalid_vertex_examples = []

    if region_data["vertices"] is not None:
        vertices = region_data["vertices"]
        n_vertices = len(vertices)
        vertices_to_check = vertices[:max_vertices]

        for i, v in enumerate(vertices_to_check):
            fan_votes = np.array(v, dtype=np.float64)
            result = verify_point_core(
                fan_votes, contestants, judge_scores, eliminated, placements, season, is_final
            )
            if result.valid:
                n_vertices_valid += 1
            elif len(invalid_vertex_examples) < 5:
                invalid_vertex_examples.append({
                    "index": i,
                    "predicted": result.predicted_eliminated,
                })

    # Check bounds
    bounds_valid = True
    if region_data["dim_bounds"] is not None:
        dim_bounds = np.array(region_data["dim_bounds"], dtype=np.float64)
        if np.any(dim_bounds[:, 0] < -1e-9):
            bounds_valid = False
        if np.any(dim_bounds[:, 1] > 1 + 1e-9):
            bounds_valid = False

    n_elim = len(eliminated)
    premise_type = get_premise_type(season, n_elim, is_final)

    return {
        "season": season,
        "week": week,
        "is_final": is_final,
        "premise_type": premise_type.name,
        "centroid_valid": centroid_valid,
        "centroid_margin": centroid_margin,
        "centroid_predicted": centroid_predicted,
        "representative_valid": representative_valid,
        "actual_eliminated": list(eliminated),
        "n_vertices": n_vertices,
        "n_vertices_valid": n_vertices_valid,
        "n_vertices_checked": min(n_vertices, max_vertices),
        "bounds_valid": bounds_valid,
        "invalid_vertex_examples": invalid_vertex_examples,
    }


# =============================================================================
# Main Parallel Verification
# =============================================================================

@dataclass
class RegionVerification:
    """Result of verifying an entire region."""
    season: int
    week: int
    is_final: bool
    premise_type: str

    centroid_valid: Optional[bool]
    centroid_margin: Optional[float]
    centroid_predicted: Optional[List[str]]
    representative_valid: Optional[bool]  # nearest-to-mean from finalization
    actual_eliminated: List[str]

    n_vertices: int
    n_vertices_valid: int
    n_vertices_checked: int
    vertices_all_valid: bool

    bounds_valid: bool
    invalid_vertex_examples: List[Dict]


def verify_all_regions_parallel(
    regions: List[Region],
    events: Dict[Tuple[int, int], Event],
    check_vertices: bool = True,
    max_vertices: int = 1000,
    verbose: bool = False,
) -> List[RegionVerification]:
    """
    Verify all regions in parallel using configured number of workers.
    """
    cfg = get_config()
    n_workers = cfg.parallel.n_workers
    batch_size = cfg.parallel.batch_size

    # Serialize events for workers
    events_serialized = {
        k: v.to_serializable() for k, v in events.items()
    }

    # Serialize regions for workers
    region_tasks = [
        (r.to_serializable(), max_vertices if check_vertices else 0, batch_size)
        for r in regions
    ]

    results = []

    if n_workers <= 1 or len(regions) <= 2:
        # Sequential execution for small workloads
        _init_worker(events_serialized)
        for task in region_tasks:
            result_dict = _verify_region_task(task)
            results.append(result_dict)
    else:
        # Parallel execution
        log.info(f"Verifying {len(regions)} regions using {n_workers} workers...")

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(events_serialized,)
        ) as executor:
            futures = [executor.submit(_verify_region_task, task) for task in region_tasks]

            for future in as_completed(futures):
                result_dict = future.result()
                results.append(result_dict)

                if verbose:
                    s, w = result_dict["season"], result_dict["week"]
                    valid = result_dict["centroid_valid"]
                    status = "PASS" if valid else ("FAIL" if valid is False else "N/A")
                    margin = result_dict.get("centroid_margin", 0) or 0
                    log.info(f"S{s}W{w}: {status} (margin={margin:.4f})")

    # Sort by season, week for consistent output
    results.sort(key=lambda r: (r["season"], r["week"]))

    # Convert to RegionVerification objects
    verifications = []
    for r in results:
        if "error" in r:
            continue

        v = RegionVerification(
            season=r["season"],
            week=r["week"],
            is_final=r["is_final"],
            premise_type=r["premise_type"],
            centroid_valid=r["centroid_valid"],
            centroid_margin=r["centroid_margin"],
            centroid_predicted=r["centroid_predicted"],
            representative_valid=r.get("representative_valid"),
            actual_eliminated=r["actual_eliminated"],
            n_vertices=r["n_vertices"],
            n_vertices_valid=r["n_vertices_valid"],
            n_vertices_checked=r["n_vertices_checked"],
            vertices_all_valid=(r["n_vertices_valid"] == r["n_vertices_checked"]) if r["n_vertices"] > 0 else True,
            bounds_valid=r["bounds_valid"],
            invalid_vertex_examples=r["invalid_vertex_examples"],
        )
        verifications.append(v)

    return verifications


# =============================================================================
# Detailed Analysis (Single Region)
# =============================================================================

def analyze_point_detail(
    fan_votes: np.ndarray,
    event: Event,
) -> Dict[str, Any]:
    """Provide detailed analysis of why a point is valid/invalid."""
    n_elim = len(event.eliminated)
    premise_type = get_premise_type(event.season, n_elim, event.is_final)

    analysis = {
        "premise_type": premise_type.name,
        "contestants": event.contestants,
        "judge_scores": event.judge_scores.tolist(),
        "fan_votes": fan_votes.tolist(),
    }

    if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI, PremiseType.RR_B2,
                        PremiseType.RR_B2_MULTI, PremiseType.RR_FINAL):
        judge_ranks = scores_to_ranks(event.judge_scores)
        fan_ranks = scores_to_ranks(fan_votes)
        combined = judge_ranks + fan_ranks

        analysis["judge_ranks"] = judge_ranks.tolist()
        analysis["fan_ranks"] = fan_ranks.tolist()
        analysis["combined_rank_sum"] = combined.tolist()
        analysis["ranking_by_combined"] = [event.contestants[i] for i in np.argsort(combined)]
        analysis["worst_first"] = [event.contestants[i] for i in np.argsort(-combined)]
    else:
        judge_pct = scores_to_percent(event.judge_scores)
        fan_pct = scores_to_percent(fan_votes)
        combined = judge_pct + fan_pct

        analysis["judge_pct"] = judge_pct.tolist()
        analysis["fan_pct"] = fan_pct.tolist()
        analysis["combined_pct"] = combined.tolist()
        analysis["ranking_by_combined"] = [event.contestants[i] for i in np.argsort(-combined)]
        analysis["worst_first"] = [event.contestants[i] for i in np.argsort(combined)]

    analysis["actual_eliminated"] = list(event.eliminated)

    return analysis


# =============================================================================
# Summary Output
# =============================================================================

def print_summary(results: List[RegionVerification]) -> None:
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    total = len(results)
    centroid_valid = sum(1 for r in results if r.centroid_valid)
    centroid_invalid = sum(1 for r in results if r.centroid_valid is False)
    centroid_none = sum(1 for r in results if r.centroid_valid is None)

    vertices_all_valid = sum(1 for r in results if r.vertices_all_valid and r.n_vertices > 0)
    vertices_some_invalid = sum(1 for r in results if not r.vertices_all_valid and r.n_vertices > 0)
    total_vertices = sum(r.n_vertices_checked for r in results)
    total_vertices_valid = sum(r.n_vertices_valid for r in results)

    bounds_valid = sum(1 for r in results if r.bounds_valid)

    print(f"\nTotal regions verified: {total}")
    print(f"\nCentroid verification:")
    print(f"  Valid:   {centroid_valid}/{total}")
    print(f"  Invalid: {centroid_invalid}/{total}")
    print(f"  No centroid: {centroid_none}/{total}")

    # Representative point (from finalization)
    rep_valid = sum(1 for r in results if r.representative_valid is True)
    rep_invalid = sum(1 for r in results if r.representative_valid is False)
    rep_none = sum(1 for r in results if r.representative_valid is None)
    if rep_valid + rep_invalid > 0:
        print(f"\nRepresentative point verification (nearest-to-mean):")
        print(f"  Valid:   {rep_valid}/{total}")
        print(f"  Invalid: {rep_invalid}/{total}")
        print(f"  No rep point: {rep_none}/{total}")

    if any(r.n_vertices > 0 for r in results):
        print(f"\nVertex verification:")
        print(f"  Total checked: {total_vertices}")
        print(f"  Total valid:   {total_vertices_valid} ({100*total_vertices_valid/total_vertices:.1f}%)" if total_vertices > 0 else "")
        print(f"  Regions with all valid:   {vertices_all_valid}")
        print(f"  Regions with some invalid: {vertices_some_invalid}")

    print(f"\nBounds verification:")
    print(f"  Valid: {bounds_valid}/{total}")

    # Group by premise type
    by_premise = defaultdict(list)
    for r in results:
        by_premise[r.premise_type].append(r)

    print(f"\nBy premise type:")
    for premise, premise_results in sorted(by_premise.items()):
        valid = sum(1 for r in premise_results if r.centroid_valid)
        print(f"  {premise}: {valid}/{len(premise_results)} centroids valid")

    # List failures
    failures = [r for r in results if r.centroid_valid is False]
    if failures:
        print(f"\nFailed regions ({len(failures)}):")
        for r in failures:
            print(f"  Season {r.season} Week {r.week} ({r.premise_type})")
            print(f"    Predicted: {r.centroid_predicted}")
            print(f"    Actual:    {r.actual_eliminated}")

    # Margin statistics
    margins = [r.centroid_margin for r in results if r.centroid_margin is not None]
    if margins:
        print(f"\nMargin statistics (valid regions):")
        print(f"  Min:  {min(margins):.6f}")
        print(f"  Max:  {max(margins):.6f}")
        print(f"  Mean: {np.mean(margins):.6f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify region centroids against elimination constraints")
    parser.add_argument("--regions", "-r",
                        default=str(DATA_DIR / "regions-finalized.json"),
                        help="Regions JSON file to verify (default: regions-finalized.json)")
    parser.add_argument("--events", "-e",
                        default=str(DATA_DIR / "events.json"),
                        help="Events JSON file with judge scores and eliminations")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to verify (default: all in file)")
    parser.add_argument("--no-vertices", action="store_true",
                        help="Skip vertex verification (faster, centroid only)")
    parser.add_argument("--max-vertices", type=int, default=1000,
                        help="Max vertices to check per region (default: 1000)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for each region")
    parser.add_argument("--detail", "-d", type=str, default=None,
                        help="Show detailed analysis for specific region (format: season,week)")
    args = parser.parse_args()

    cfg = get_config()
    log.info(f"Using {cfg.parallel.n_workers} workers for parallel verification")

    # Load data
    log.info(f"Loading events from {args.events}")
    events = load_events(Path(args.events))
    log.info(f"Loaded {len(events)} events")

    log.info(f"Loading regions from {args.regions}")
    regions = load_regions(Path(args.regions))
    log.info(f"Loaded {len(regions)} regions")

    # Filter by season if specified
    if args.seasons:
        season_filter = set(int(s.strip()) for s in args.seasons.split(","))
        regions = [r for r in regions if r.season in season_filter]
        log.info(f"Filtered to {len(regions)} regions in seasons {season_filter}")

    # Detailed analysis for specific region
    if args.detail:
        season, week = map(int, args.detail.split(","))
        region = next((r for r in regions if r.season == season and r.week == week), None)
        event = events.get((season, week))

        if region is None or event is None:
            log.error(f"Region or event not found for Season {season} Week {week}")
            return

        print(f"\n{'='*70}")
        print(f"DETAILED ANALYSIS: Season {season} Week {week}")
        print(f"{'='*70}")

        if region.centroid is not None:
            analysis = analyze_point_detail(region.centroid, event)
            print(f"\nPremise: {analysis['premise_type']}")
            print(f"\nContestants: {analysis['contestants']}")
            print(f"Actual eliminated: {analysis['actual_eliminated']}")

            print(f"\nJudge scores: {analysis['judge_scores']}")
            print(f"Fan votes (centroid): {[f'{v:.4f}' for v in analysis['fan_votes']]}")

            if 'judge_ranks' in analysis:
                print(f"\nJudge ranks: {analysis['judge_ranks']}")
                print(f"Fan ranks: {analysis['fan_ranks']}")
                print(f"Combined rank sum: {analysis['combined_rank_sum']}")
            else:
                print(f"\nJudge %: {[f'{v:.4f}' for v in analysis['judge_pct']]}")
                print(f"Fan %: {[f'{v:.4f}' for v in analysis['fan_pct']]}")
                print(f"Combined %: {[f'{v:.4f}' for v in analysis['combined_pct']]}")

            print(f"\nRanking (best to worst): {analysis['ranking_by_combined']}")
            print(f"Worst first: {analysis['worst_first']}")

            result = verify_point_core(
                region.centroid, event.contestants, event.judge_scores,
                event.eliminated, event.placements, event.season, event.is_final
            )
            print(f"\nVerification: {'PASS' if result.valid else 'FAIL'}")
            print(f"Predicted eliminated: {result.predicted_eliminated}")
            print(f"Margin: {result.margin:.6f}")

        return

    # Verify all regions in parallel
    results = verify_all_regions_parallel(
        regions, events,
        check_vertices=not args.no_vertices,
        max_vertices=args.max_vertices,
        verbose=args.verbose,
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
