#!/usr/bin/env python3
"""
Finalize regions by sampling from convex hull and verifying each point.

Takes forward-projected regions, samples points uniformly from the convex hull,
verifies each against premise constraints, and produces a validated point cloud
with comprehensive statistics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from numba import njit

from config import get_config
from structures import WeekRegion, load_regions, save_results
from geometry import simplex_volume

# Load simple-models/premises.py directly for full access (matches verify.py)
import importlib.util
_premises_path = Path(__file__).parent.parent / "simple-models" / "premises.py"
_spec = importlib.util.spec_from_file_location("simple_premises", _premises_path)
_premises = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_premises)

PremiseType = _premises.PremiseType
get_premise_type = _premises.get_premise_type
validate = _premises.validate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Sampling from convex hull
# =============================================================================

def sample_from_hull(vertices: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample points uniformly from the convex hull of vertices.

    Uses the simplex method: sample random convex combinations of vertices.
    This is uniform over the vertices but not perfectly uniform over the hull;
    for a true uniform sample over the hull we'd need rejection sampling,
    but this is a good approximation.
    """
    n_verts = len(vertices)
    if n_verts == 0:
        return np.empty((0, vertices.shape[1] if len(vertices.shape) > 1 else 0))

    if n_verts == 1:
        return np.tile(vertices[0], (n_samples, 1))

    # Sample from Dirichlet to get convex combinations
    # Using alpha=1 gives uniform distribution over the simplex of weights
    weights = rng.dirichlet(np.ones(n_verts), size=n_samples)

    # Compute convex combinations
    samples = weights @ vertices

    return samples


def sample_from_simplex(n_dims: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from the probability simplex."""
    return rng.dirichlet(np.ones(n_dims), size=n_samples)


# =============================================================================
# Event loading for verification
# =============================================================================

@dataclass
class EventInfo:
    """Minimal event info needed for verification."""
    season: int
    week: int
    is_final: bool
    contestants: List[str]
    judge_scores: np.ndarray
    eliminated: Set[str]
    placements: Optional[np.ndarray]
    premise_type: PremiseType


def load_events_dict() -> Dict[Tuple[int, int, bool], EventInfo]:
    """Load events and index by (season, week, is_final)."""
    events_path = DATA_DIR / "events.json"

    if not events_path.exists():
        log.warning(f"Events file not found at {events_path}, regenerating...")
        from events import load_events, save_events
        events = load_events()
        save_events(events)

    with open(events_path) as f:
        events_data = json.load(f)

    events_dict = {}
    for e in events_data:
        key = (e["season"], e["week"], e.get("is_final", False))
        premise = get_premise_type(e["season"], len(e.get("eliminated", [])), e.get("is_final", False))

        placements = None
        if "placements" in e and e["placements"]:
            placements = np.array(e["placements"])

        events_dict[key] = EventInfo(
            season=e["season"],
            week=e["week"],
            is_final=e.get("is_final", False),
            contestants=e["contestants"],
            judge_scores=np.array(e["judge_scores"]),
            eliminated=set(e.get("eliminated", [])),
            placements=placements,
            premise_type=premise,
        )

    return events_dict


def verify_point(point: np.ndarray, event: EventInfo) -> bool:
    """Verify a single point against premise constraints using same logic as verify.py."""
    if event.is_final:
        if event.placements is None:
            return False
        return validate(
            event.premise_type,
            event.contestants,
            event.judge_scores,
            point,
            actual_placements=event.placements,
        )
    else:
        return validate(
            event.premise_type,
            event.contestants,
            event.judge_scores,
            point,
            actual_eliminated=event.eliminated,
        )


# =============================================================================
# Statistics computation
# =============================================================================

def compute_cloud_statistics(
    points: np.ndarray,
    contestants: List[str],
    precision: int = 10,
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a point cloud.

    Returns dict with:
    - n_points: number of points in cloud
    - mean: mean vector
    - nearest_to_mean: point nearest to mean (more representative than mean itself)
    - nearest_to_mean_distance: distance from mean to nearest point
    - std: standard deviation per coordinate
    - dim_bounds: min/max/delta per coordinate with exact vectors
    """
    if len(points) == 0:
        return {
            "n_points": 0,
            "mean": None,
            "nearest_to_mean": None,
            "nearest_to_mean_distance": None,
            "std": None,
            "dim_bounds": None,
        }

    n_points, n_dims = points.shape
    mean = points.mean(axis=0)
    std = points.std(axis=0)

    # Find point nearest to mean
    distances = np.linalg.norm(points - mean, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_to_mean = points[nearest_idx]
    nearest_distance = distances[nearest_idx]

    # Per-dimension bounds with extrema vectors
    dim_bounds = []
    for i in range(n_dims):
        col = points[:, i]
        min_idx = np.argmin(col)
        max_idx = np.argmax(col)

        dim_bounds.append({
            "contestant": contestants[i] if i < len(contestants) else f"dim_{i}",
            "min": round(float(col[min_idx]), precision),
            "max": round(float(col[max_idx]), precision),
            "delta": round(float(col[max_idx] - col[min_idx]), precision),
            "min_vector": [round(float(x), precision) for x in points[min_idx]],
            "max_vector": [round(float(x), precision) for x in points[max_idx]],
        })

    return {
        "n_points": n_points,
        "mean": [round(float(x), precision) for x in mean],
        "nearest_to_mean": [round(float(x), precision) for x in nearest_to_mean],
        "nearest_to_mean_distance": round(float(nearest_distance), precision),
        "std": [round(float(x), precision) for x in std],
        "dim_bounds": dim_bounds,
    }


def compute_extreme_statistics(
    points: np.ndarray,
    precision: int = 10,
) -> Dict[str, Any]:
    """
    Compute extreme value statistics.

    - diameter: max pairwise distance
    - diameter_points: the two points achieving max distance
    - centroid_distance: max distance from any point to centroid
    - volume_estimate: approximate volume via sampling density
    """
    if len(points) < 2:
        return {
            "diameter": 0.0,
            "diameter_points": None,
            "max_centroid_distance": 0.0,
            "farthest_from_centroid": None,
        }

    from scipy.spatial.distance import pdist, squareform

    # Diameter
    dists = pdist(points)
    if len(dists) > 0:
        diameter = float(dists.max())
        # Find which pair
        dist_matrix = squareform(dists)
        i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        diameter_points = [
            [round(float(x), precision) for x in points[i]],
            [round(float(x), precision) for x in points[j]],
        ]
    else:
        diameter = 0.0
        diameter_points = None

    # Farthest from centroid
    centroid = points.mean(axis=0)
    centroid_dists = np.linalg.norm(points - centroid, axis=1)
    farthest_idx = np.argmax(centroid_dists)

    return {
        "diameter": round(diameter, precision),
        "diameter_points": diameter_points,
        "max_centroid_distance": round(float(centroid_dists[farthest_idx]), precision),
        "farthest_from_centroid": [round(float(x), precision) for x in points[farthest_idx]],
    }


# =============================================================================
# Main finalization logic
# =============================================================================

def finalize_region(
    region: WeekRegion,
    event: EventInfo,
    n_hull_samples: int,
    n_simplex_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Finalize a single region by sampling and verifying.

    Returns updated region dict with point cloud and statistics.
    """
    cfg = get_config()
    p = cfg.output.precision
    rng = np.random.default_rng(seed)

    result = json.loads(json.dumps(region.raw_data))  # deep copy
    n = region.n_contestants

    # Initialize finalization section
    result["finalization"] = {
        "hull_samples": n_hull_samples,
        "simplex_samples": n_simplex_samples,
        "seed": seed,
    }

    # Check if we have vertices to sample from
    if region.vertices is None or len(region.vertices) == 0:
        log.warning(f"  No vertices for S{region.season}W{region.week}")
        result["finalization"]["status"] = "no_vertices"
        result["finalization"]["point_cloud"] = []
        result["finalization"]["cloud_statistics"] = compute_cloud_statistics(np.empty((0, n)), region.contestants, p)
        result["finalization"]["extreme_statistics"] = compute_extreme_statistics(np.empty((0, n)), p)
        result["finalization"]["hull_acceptance"] = 0.0
        result["finalization"]["simplex_accuracy"] = 0.0
        return result

    # Sample from hull and verify
    log.info(f"  Sampling {n_hull_samples} points from hull...")
    hull_samples = sample_from_hull(region.vertices, n_hull_samples, rng)

    # Verify each point
    valid_mask = np.array([verify_point(pt, event) for pt in hull_samples])
    valid_points = hull_samples[valid_mask]
    hull_acceptance = valid_mask.sum() / len(hull_samples) if len(hull_samples) > 0 else 0.0

    log.info(f"  Hull acceptance: {valid_mask.sum()}/{len(hull_samples)} ({hull_acceptance:.1%})")

    # Sample from full simplex for accuracy estimate
    log.info(f"  Sampling {n_simplex_samples} points from simplex...")
    simplex_samples = sample_from_simplex(n, n_simplex_samples, rng)
    simplex_valid_mask = np.array([verify_point(pt, event) for pt in simplex_samples])
    simplex_accuracy = simplex_valid_mask.sum() / len(simplex_samples) if len(simplex_samples) > 0 else 0.0

    log.info(f"  Simplex accuracy: {simplex_valid_mask.sum()}/{len(simplex_samples)} ({simplex_accuracy:.4%})")

    # Compute statistics
    cloud_stats = compute_cloud_statistics(valid_points, region.contestants, p)
    extreme_stats = compute_extreme_statistics(valid_points, p)

    # Store results
    result["finalization"]["status"] = "success"
    result["finalization"]["hull_acceptance"] = round(float(hull_acceptance), p)
    result["finalization"]["simplex_accuracy"] = round(float(simplex_accuracy), p + 4)
    result["finalization"]["cloud_statistics"] = cloud_stats
    result["finalization"]["extreme_statistics"] = extreme_stats

    # Store the entire point cloud
    result["finalization"]["point_cloud"] = [
        [round(float(x), p) for x in pt]
        for pt in valid_points
    ]

    # Update region's representative point to nearest-to-mean
    if cloud_stats["nearest_to_mean"] is not None:
        result["region"]["representative_point"] = cloud_stats["nearest_to_mean"]

    return result


def finalize_season(
    regions: List[WeekRegion],
    events_dict: Dict[Tuple[int, int, bool], EventInfo],
    n_hull_samples: int,
    n_simplex_samples: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Finalize all regions for a single season."""
    results = []

    for i, region in enumerate(sorted(regions, key=lambda r: (r.is_final, r.week))):
        key = (region.season, region.week, region.is_final)
        event = events_dict.get(key)

        if event is None:
            log.warning(f"  No event found for S{region.season}W{region.week}")
            result = json.loads(json.dumps(region.raw_data))
            result["finalization"] = {"status": "no_event"}
            results.append(result)
            continue

        tag = f"W{region.week}" + (" (final)" if region.is_final else "")
        log.info(f"  {tag}: {region.n_contestants} contestants")

        result = finalize_region(
            region, event,
            n_hull_samples=n_hull_samples,
            n_simplex_samples=n_simplex_samples,
            seed=base_seed + i,
        )
        results.append(result)

    return results


def _process_season_worker(season: int, shared_data: Dict[str, Any]) -> Tuple[List[Dict], List[str]]:
    """
    Worker function to process a single season.
    Called in parallel for each season.
    """
    from structures import WeekRegion

    regions_data = shared_data["by_season"][season]
    events_dict = shared_data["events_dict"]
    n_hull_samples = shared_data["n_hull_samples"]
    n_simplex_samples = shared_data["n_simplex_samples"]
    base_seed = shared_data["base_seed"]

    # Reconstruct WeekRegion objects
    regions = [WeekRegion.from_dict(r) for r in regions_data]

    # Reconstruct events_dict with proper types
    events_dict_typed = {}
    for key_str, event_data in events_dict.items():
        # Key was serialized as string, parse it back
        # Format: "(season, week, is_final)"
        key = eval(key_str)  # Safe here since we control the format
        events_dict_typed[key] = EventInfo(
            season=event_data["season"],
            week=event_data["week"],
            is_final=event_data["is_final"],
            contestants=event_data["contestants"],
            judge_scores=np.array(event_data["judge_scores"]),
            eliminated=set(event_data["eliminated"]),
            placements=np.array(event_data["placements"]) if event_data["placements"] else None,
            premise_type=PremiseType[event_data["premise_type"]],
        )

    results = finalize_season(
        regions,
        events_dict_typed,
        n_hull_samples=n_hull_samples,
        n_simplex_samples=n_simplex_samples,
        base_seed=base_seed + season * 100,
    )

    return results, []


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(description="Finalize regions with verified point clouds")
    parser.add_argument("--input", "-i", default=str(DATA_DIR / "regions-forwardprojected.json"),
                        help="Input regions JSON path")
    parser.add_argument("--output", "-o", default=str(DATA_DIR / "regions-finalized.json"),
                        help="Output path")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to process")
    parser.add_argument("--hull-samples", type=int, default=10000,
                        help="Number of samples from hull per region")
    parser.add_argument("--simplex-samples", type=int, default=100000,
                        help="Number of samples from simplex for accuracy estimate")
    parser.add_argument("--seed", type=int, default=cfg.sampling.seed,
                        help="Random seed")
    parser.add_argument("--sequential", action="store_true",
                        help="Process seasons sequentially")
    args = parser.parse_args()

    # Load regions
    log.info(f"Loading regions from {args.input}")
    regions = load_regions(Path(args.input))
    log.info(f"Loaded {len(regions)} regions")

    # Load events
    log.info("Loading events for verification...")
    events_dict = load_events_dict()
    log.info(f"Loaded {len(events_dict)} events")

    # Group by season
    by_season: Dict[int, List[WeekRegion]] = defaultdict(list)
    for r in regions:
        by_season[r.season].append(r)

    # Filter seasons if specified
    if args.seasons:
        season_filter = set(int(s.strip()) for s in args.seasons.split(","))
        by_season = {s: rs for s, rs in by_season.items() if s in season_filter}

    n_seasons = len(by_season)
    log.info(f"Processing {n_seasons} seasons")

    all_results = []

    if args.sequential or n_seasons <= 1:
        # Sequential processing
        for season in sorted(by_season.keys()):
            log.info(f"\n{'='*60}")
            log.info(f"Season {season}")
            log.info(f"{'='*60}")

            season_results = finalize_season(
                by_season[season],
                events_dict,
                n_hull_samples=args.hull_samples,
                n_simplex_samples=args.simplex_samples,
                base_seed=args.seed + season * 100,
            )
            all_results.extend(season_results)
    else:
        # Parallel processing
        from parallel import run_parallel_seasons

        # Serialize for workers
        by_season_serialized = {
            s: [r.raw_data for r in rs]
            for s, rs in by_season.items()
        }

        # Serialize events_dict (convert keys to strings, EventInfo to dicts)
        events_dict_serialized = {}
        for key, event in events_dict.items():
            key_str = str(key)
            events_dict_serialized[key_str] = {
                "season": event.season,
                "week": event.week,
                "is_final": event.is_final,
                "contestants": event.contestants,
                "judge_scores": event.judge_scores.tolist(),
                "eliminated": list(event.eliminated),
                "placements": event.placements.tolist() if event.placements is not None else None,
                "premise_type": event.premise_type.name,
            }

        shared_data = {
            "by_season": by_season_serialized,
            "events_dict": events_dict_serialized,
            "n_hull_samples": args.hull_samples,
            "n_simplex_samples": args.simplex_samples,
            "base_seed": args.seed,
        }

        log.info(f"Using {cfg.parallel.n_workers} workers for parallel processing")

        season_results = run_parallel_seasons(
            seasons=sorted(by_season.keys()),
            process_fn=_process_season_worker,
            shared_data=shared_data,
            task_name="Finalizing",
        )

        # Collect in season order
        for season in sorted(season_results.keys()):
            all_results.extend(season_results[season])

    # Save results
    log.info(f"\nSaving {len(all_results)} regions to {args.output}")
    save_results(all_results, Path(args.output))

    # Summary
    print("\n" + "=" * 60)
    print("FINALIZATION SUMMARY")
    print("=" * 60)

    successful = [r for r in all_results if r.get("finalization", {}).get("status") == "success"]

    if successful:
        hull_acceptances = [r["finalization"]["hull_acceptance"] for r in successful]
        simplex_accuracies = [r["finalization"]["simplex_accuracy"] for r in successful]
        cloud_sizes = [r["finalization"]["cloud_statistics"]["n_points"] for r in successful]

        print(f"Total regions: {len(all_results)}")
        print(f"Successfully finalized: {len(successful)}")
        print(f"\nHull acceptance rate:")
        print(f"  Mean: {np.mean(hull_acceptances):.1%}")
        print(f"  Min:  {np.min(hull_acceptances):.1%}")
        print(f"  Max:  {np.max(hull_acceptances):.1%}")
        print(f"\nSimplex accuracy (fraction of full simplex that's valid):")
        print(f"  Mean: {np.mean(simplex_accuracies):.4%}")
        print(f"  Min:  {np.min(simplex_accuracies):.4%}")
        print(f"  Max:  {np.max(simplex_accuracies):.4%}")
        print(f"\nPoint cloud sizes:")
        print(f"  Total points: {sum(cloud_sizes):,}")
        print(f"  Mean per region: {np.mean(cloud_sizes):.0f}")
        print(f"  Min: {np.min(cloud_sizes)}")
        print(f"  Max: {np.max(cloud_sizes)}")

    # Warn about low hull acceptance
    low_acceptance = [r for r in successful if r["finalization"]["hull_acceptance"] < 0.5]
    if low_acceptance:
        print(f"\n*** WARNING: {len(low_acceptance)} regions with <50% hull acceptance ***")
        for r in low_acceptance[:5]:
            e = r["event"]
            f = r["finalization"]
            print(f"  S{e['season']}W{e['week']}: {f['hull_acceptance']:.1%} acceptance, {f['cloud_statistics']['n_points']} valid points")


if __name__ == "__main__":
    main()
