#!/usr/bin/env python3
"""
Backprojection: Constrain earlier weeks by later weeks.

Under fixed voter preferences, week N+1's vote distribution constrains
what week N could have been. When contestant X is eliminated, X's voters
redistribute to their next preferred contestant still in the race.

This script filters week N's valid region to only points that can project
forward to land within week N+1's region.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import logging
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from math import factorial

from config import get_config


def simplex_volume(n: int) -> float:
    """Volume of the (n-1)-simplex in projected coordinates."""
    if n <= 1:
        return 1.0
    return 1.0 / factorial(n - 1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class WeekRegion:
    """Parsed region data for a single week."""
    season: int
    week: int
    is_final: bool
    contestants: List[str]
    n_contestants: int
    premise_type: str
    n_valid: int
    vertices: Optional[np.ndarray]  # shape (n_vertices, n_contestants)
    dim_bounds: Optional[np.ndarray]  # shape (n_contestants, 2) for [lo, hi]
    centroid: Optional[np.ndarray]
    raw_data: Dict[str, Any]  # original JSON data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeekRegion":
        """Parse from JSON dict."""
        event = d["event"]
        region = d["region"]

        vertices = None
        if "vertices" in region and region["vertices"]:
            vertices = np.array(region["vertices"])

        dim_bounds = None
        if "dim_bounds" in region and region["dim_bounds"]:
            dim_bounds = np.array(region["dim_bounds"])

        centroid = None
        if "centroid" in region and region["centroid"]:
            centroid = np.array(region["centroid"])

        return cls(
            season=event["season"],
            week=event["week"],
            is_final=event["is_final"],
            contestants=event["contestants"],
            n_contestants=event["n_contestants"],
            premise_type=event["premise_type"],
            n_valid=d["sampling"]["n_valid"],
            vertices=vertices,
            dim_bounds=dim_bounds,
            centroid=centroid,
            raw_data=d,
        )


# =============================================================================
# Core projection logic
# =============================================================================

def find_eliminated(week_n: WeekRegion, week_n1: WeekRegion) -> List[str]:
    """Find contestants eliminated between week N and week N+1."""
    return [c for c in week_n.contestants if c not in week_n1.contestants]


def get_contestant_mapping(
    week_n: WeekRegion,
    week_n1: WeekRegion,
) -> Tuple[List[int], List[int]]:
    """
    Get index mappings between weeks.

    Returns:
        survivors_n: indices in week_n of contestants still in week_n1
        survivors_n1: corresponding indices in week_n1
    """
    survivors_n = []
    survivors_n1 = []

    for i, name in enumerate(week_n.contestants):
        if name in week_n1.contestants:
            survivors_n.append(i)
            survivors_n1.append(week_n1.contestants.index(name))

    return survivors_n, survivors_n1


def check_point_projects_lp(
    point: np.ndarray,
    eliminated_indices: List[int],
    survivor_indices_n: List[int],
    survivor_indices_n1: List[int],
    target_bounds: np.ndarray,
    tolerance: float = 1e-9,
    padding: float = 0.001,
) -> bool:
    """
    Check if point can project to target region via LP.

    The redistribution problem:
    - Eliminated contestants' votes redistribute to survivors
    - Find if any valid redistribution lands inside target_bounds

    LP formulation:
    Variables: redistribution fractions α_ij (from eliminated i to survivor j)
    Constraints:
        - sum_j α_ij = 1 for each eliminated i (all votes must go somewhere)
        - α_ij >= 0
        - lo_j <= survivor_vote_j + sum_i (α_ij * eliminated_vote_i) <= hi_j

    Args:
        point: vote distribution in week N (sums to 1)
        eliminated_indices: indices of eliminated contestants in week N
        survivor_indices_n: indices of survivors in week N
        survivor_indices_n1: corresponding indices in week N+1
        target_bounds: (n_survivors, 2) array of [lo, hi] bounds in week N+1
        tolerance: LP solver tolerance
        padding: small padding on bounds for numerical stability

    Returns:
        True if LP is feasible (point can project to target)
    """
    n_elim = len(eliminated_indices)
    n_surv = len(survivor_indices_n)

    if n_elim == 0:
        # No elimination, check if survivors are directly in bounds
        for j, (idx_n, idx_n1) in enumerate(zip(survivor_indices_n, survivor_indices_n1)):
            lo, hi = target_bounds[idx_n1]
            if point[idx_n] < lo - padding or point[idx_n] > hi + padding:
                return False
        return True

    # Variables: α_ij for i in [0, n_elim), j in [0, n_surv)
    # Flattened: α[i * n_surv + j]
    n_vars = n_elim * n_surv

    # Objective: we just want feasibility, so minimize 0
    c = np.zeros(n_vars)

    # Equality constraints: sum_j α_ij = 1 for each i
    A_eq = np.zeros((n_elim, n_vars))
    for i in range(n_elim):
        for j in range(n_surv):
            A_eq[i, i * n_surv + j] = 1.0
    b_eq = np.ones(n_elim)

    # Inequality constraints for bounds
    # For each survivor j: lo_j <= base_j + sum_i α_ij * elim_i <= hi_j
    # Rewritten as:
    #   sum_i α_ij * elim_i >= lo_j - base_j
    #   sum_i α_ij * elim_i <= hi_j - base_j

    A_ub_list = []
    b_ub_list = []

    for j, (idx_n, idx_n1) in enumerate(zip(survivor_indices_n, survivor_indices_n1)):
        base_vote = point[idx_n]
        lo, hi = target_bounds[idx_n1]

        # Apply padding
        lo = lo - padding
        hi = hi + padding

        # Coefficients for sum_i α_ij * elim_i
        coeffs = np.zeros(n_vars)
        for i, elim_idx in enumerate(eliminated_indices):
            coeffs[i * n_surv + j] = point[elim_idx]

        # Upper bound: sum <= hi - base  =>  sum - hi + base <= 0
        A_ub_list.append(coeffs.copy())
        b_ub_list.append(hi - base_vote)

        # Lower bound: sum >= lo - base  =>  -sum <= -(lo - base)
        A_ub_list.append(-coeffs)
        b_ub_list.append(-(lo - base_vote))

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None

    # Variable bounds: 0 <= α_ij <= 1
    bounds = [(0, 1) for _ in range(n_vars)]

    # Solve LP
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    return result.success


def check_point_projects_sample(
    point: np.ndarray,
    eliminated_indices: List[int],
    survivor_indices_n: List[int],
    survivor_indices_n1: List[int],
    target_bounds: np.ndarray,
    n_samples: int = 1000,
    padding: float = 0.001,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """
    Check if point can project to target region via sampling.

    Sample random redistributions and check if any lands in bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_elim = len(eliminated_indices)
    n_surv = len(survivor_indices_n)

    if n_elim == 0:
        for j, (idx_n, idx_n1) in enumerate(zip(survivor_indices_n, survivor_indices_n1)):
            lo, hi = target_bounds[idx_n1]
            if point[idx_n] < lo - padding or point[idx_n] > hi + padding:
                return False
        return True

    # Sample redistributions from product of simplices
    for _ in range(n_samples):
        # For each eliminated contestant, sample redistribution to survivors
        projected = np.zeros(len(target_bounds))

        # Start with survivor base votes
        for j, (idx_n, idx_n1) in enumerate(zip(survivor_indices_n, survivor_indices_n1)):
            projected[idx_n1] = point[idx_n]

        # Add redistributed votes
        for i, elim_idx in enumerate(eliminated_indices):
            elim_vote = point[elim_idx]
            # Sample from simplex
            redist = rng.dirichlet(np.ones(n_surv))
            for j, idx_n1 in enumerate(survivor_indices_n1):
                projected[idx_n1] += redist[j] * elim_vote

        # Check bounds
        in_bounds = True
        for j in range(len(target_bounds)):
            lo, hi = target_bounds[j]
            if projected[j] < lo - padding or projected[j] > hi + padding:
                in_bounds = False
                break

        if in_bounds:
            return True

    return False


# =============================================================================
# Region filtering
# =============================================================================

def filter_region_by_projection(
    week_n: WeekRegion,
    week_n1: WeekRegion,
    method: str = "lp",
    tolerance: float = 1e-9,
    padding: float = 0.001,
    n_samples: int = 1000,
) -> Tuple[np.ndarray, int]:
    """
    Filter week N's vertices to those that can project to week N+1.

    Returns:
        filtered_vertices: vertices that passed the projection test
        n_filtered: count of filtered vertices
    """
    if week_n.vertices is None or len(week_n.vertices) == 0:
        return np.array([]), 0

    if week_n1.dim_bounds is None:
        log.warning(f"  Week {week_n1.week} has no dim_bounds, skipping constraint")
        return week_n.vertices, len(week_n.vertices)

    # Get mappings
    eliminated = find_eliminated(week_n, week_n1)
    eliminated_indices = [week_n.contestants.index(e) for e in eliminated]
    survivor_indices_n, survivor_indices_n1 = get_contestant_mapping(week_n, week_n1)

    log.info(f"  Eliminated: {eliminated}")
    log.info(f"  Checking {len(week_n.vertices)} vertices...")

    # Choose method
    if method == "lp":
        check_fn = lambda p: check_point_projects_lp(
            p,
            eliminated_indices,
            survivor_indices_n,
            survivor_indices_n1,
            week_n1.dim_bounds,
            tolerance=tolerance,
            padding=padding,
        )
    else:
        rng = np.random.default_rng(42)
        check_fn = lambda p: check_point_projects_sample(
            p,
            eliminated_indices,
            survivor_indices_n,
            survivor_indices_n1,
            week_n1.dim_bounds,
            n_samples=n_samples,
            padding=padding,
            rng=rng,
        )

    # Filter vertices
    valid_mask = np.array([check_fn(v) for v in week_n.vertices])
    filtered = week_n.vertices[valid_mask]

    return filtered, len(filtered)


def compute_volume(vertices: np.ndarray) -> float:
    """
    Compute convex hull volume.
    Uses exact scipy.spatial.ConvexHull for low dimensions,
    falls back to approximation for high dimensions.
    """
    from scipy.spatial import ConvexHull

    cfg = get_config()

    if len(vertices) < 2:
        return 0.0

    # Project to n-1 dims (simplex constraint)
    points = vertices[:, :-1]
    n_points, dim = points.shape

    if dim == 0 or n_points < dim + 1:
        return 0.0

    # Use approximation for high dimensions (ConvexHull is exponential)
    if dim > cfg.hull.max_dim_exact_volume:
        return _approximate_volume(points, cfg.hull.volume_samples)

    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return _approximate_volume(points, cfg.hull.volume_samples)


def _approximate_volume_fast(vertices: np.ndarray) -> float:
    """Fast volume approximation WITHOUT building ConvexHull."""
    n_points, dim = vertices.shape
    if dim == 0 or n_points < dim + 1:
        return 0.0

    centroid = vertices.mean(axis=0)
    diffs = vertices - centroid
    avg_sq_dist = np.mean(np.sum(diffs ** 2, axis=1))
    r = np.sqrt(avg_sq_dist)
    return (r ** dim) / factorial(dim)


def _approximate_volume(vertices: np.ndarray, n_samples: int = 100000) -> float:
    """Approximate convex hull volume."""
    n_points, dim = vertices.shape
    if dim == 0 or n_points < dim + 1:
        return 0.0

    if dim > 6:
        return _approximate_volume_fast(vertices)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    widths = maxs - mins
    if np.any(widths == 0):
        return 0.0

    box_volume = np.prod(widths)

    rng = np.random.default_rng(42)
    samples = rng.uniform(mins, maxs, size=(n_samples, dim))

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        samples_h = np.hstack([samples, np.ones((n_samples, 1))])
        inside = np.all(samples_h @ hull.equations.T <= 1e-10, axis=1)
        fraction_inside = inside.sum() / n_samples
        return box_volume * fraction_inside
    except Exception:
        return _approximate_volume_fast(vertices)


def compute_diameter(vertices: np.ndarray) -> float:
    """Compute diameter (max pairwise distance) of vertices."""
    if len(vertices) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    distances = pdist(vertices)
    return float(distances.max()) if len(distances) > 0 else 0.0


def recompute_region_stats(
    vertices: np.ndarray,
    original_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """
    Recompute region statistics from filtered vertices.

    Returns:
        result: updated data dict
        filtered_volume: volume of the filtered region
    """
    result = json.loads(json.dumps(original_data))  # deep copy

    if len(vertices) == 0:
        result["region"]["has_hull"] = False
        result["region"]["n_vertices"] = 0
        result["region"]["vertices"] = []
        result["region"]["dim_bounds"] = None
        result["region"]["centroid"] = None
        result["region"]["volume"] = 0.0
        result["region"]["relative_volume"] = 0.0
        result["region"]["diameter"] = 0.0
        return result, 0.0

    cfg = get_config()
    p = cfg.output.precision

    # Compute new volume and diameter
    filtered_volume = compute_volume(vertices)
    diameter = compute_diameter(vertices)

    # Compute relative volume (compared to full simplex)
    n_contestants = vertices.shape[1]
    full_simplex_vol = simplex_volume(n_contestants)
    relative_volume = filtered_volume / full_simplex_vol if full_simplex_vol > 0 else 0.0

    # Update stats
    result["region"]["n_vertices"] = len(vertices)
    result["region"]["vertices"] = [[round(float(x), p) for x in row] for row in vertices]
    result["region"]["centroid"] = [round(float(x), p) for x in vertices.mean(axis=0)]
    result["region"]["dim_bounds"] = [
        [round(float(vertices[:, i].min()), p), round(float(vertices[:, i].max()), p)]
        for i in range(vertices.shape[1])
    ]
    result["region"]["volume"] = round(float(filtered_volume), p + 4)
    result["region"]["relative_volume"] = round(float(relative_volume), p + 4)
    result["region"]["diameter"] = round(float(diameter), p)

    return result, filtered_volume


# =============================================================================
# Main backprojection pipeline
# =============================================================================

def backproject_season(
    regions: List[WeekRegion],
    method: str = "lp",
    tolerance: float = 1e-9,
    padding: float = 0.001,
    n_samples: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Backproject constraints for a single season.

    Works backward from final week, constraining each week by the next.
    """
    # Sort by week descending (work backward)
    sorted_regions = sorted(regions, key=lambda r: (r.is_final, r.week), reverse=True)

    results = []
    next_region: Optional[WeekRegion] = None

    for region in sorted_regions:
        if next_region is None:
            # Final week or last week with data - no constraint
            log.info(f"  Week {region.week} (final/last): no forward constraint")
            result = json.loads(json.dumps(region.raw_data))  # deep copy
            result["backprojection"] = {"constrained_by": None}
            results.append(result)
        else:
            # Constrain by next week
            log.info(f"  Week {region.week} -> Week {next_region.week}")

            # Get original volume
            original_volume = region.raw_data.get("region", {}).get("volume", 0.0)

            filtered_vertices, n_filtered = filter_region_by_projection(
                region, next_region,
                method=method,
                tolerance=tolerance,
                padding=padding,
                n_samples=n_samples,
            )

            log.info(f"  Filtered: {n_filtered}/{len(region.vertices) if region.vertices is not None else 0} vertices")

            result, filtered_volume = recompute_region_stats(filtered_vertices, region.raw_data)

            # Compute volume metrics
            volume_lost = original_volume - filtered_volume
            volume_lost_pct = volume_lost / original_volume if original_volume > 0 else 0.0
            # IoU: intersection / union. Since filtered ⊆ original, intersection=filtered, union=original
            iou = filtered_volume / original_volume if original_volume > 0 else 0.0

            cfg = get_config()
            p = cfg.output.precision

            result["backprojection"] = {
                "constrained_by": next_region.week,
                "original_volume": round(float(original_volume), p + 4),
                "filtered_volume": round(float(filtered_volume), p + 4),
                "volume_lost": round(float(volume_lost), p + 4),
                "volume_lost_pct": round(float(volume_lost_pct), p),
                "iou": round(float(iou), p),
            }
            results.append(result)

        next_region = region

    # Reverse to get chronological order
    results.reverse()
    return results


def load_regions(path: Path) -> List[WeekRegion]:
    """Load and parse regions from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [WeekRegion.from_dict(d) for d in data]


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    """Save backprojected results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(description="Backproject region constraints")
    parser.add_argument("--input", "-i", default=str(DATA_DIR / "regions.json"),
                        help="Input regions.json path")
    parser.add_argument("--output", "-o", default=str(DATA_DIR / "regions-backprojected.json"),
                        help="Output path")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to process")
    parser.add_argument("--method", "-m", default=cfg.backprojection.method,
                        choices=["lp", "sample"],
                        help="Projection check method")
    args = parser.parse_args()

    # Load regions
    log.info(f"Loading regions from {args.input}")
    regions = load_regions(Path(args.input))
    log.info(f"Loaded {len(regions)} regions")

    # Group by season
    by_season: Dict[int, List[WeekRegion]] = defaultdict(list)
    for r in regions:
        by_season[r.season].append(r)

    # Filter seasons if specified
    if args.seasons:
        season_filter = set(int(s.strip()) for s in args.seasons.split(","))
        by_season = {s: rs for s, rs in by_season.items() if s in season_filter}

    log.info(f"Processing {len(by_season)} seasons")

    # Process each season
    all_results = []
    bp_cfg = cfg.backprojection

    for season in sorted(by_season.keys()):
        log.info(f"\n{'='*60}")
        log.info(f"Season {season}")
        log.info(f"{'='*60}")

        season_results = backproject_season(
            by_season[season],
            method=args.method,
            tolerance=bp_cfg.tolerance,
            padding=bp_cfg.bounds_padding,
            n_samples=bp_cfg.n_redistribution_samples,
        )
        all_results.extend(season_results)

    # Save results
    log.info(f"\nSaving {len(all_results)} regions to {args.output}")
    save_results(all_results, Path(args.output))

    # Summary
    print("\n" + "=" * 60)
    print("BACKPROJECTION SUMMARY")
    print("=" * 60)

    total_original = sum(
        r.get("backprojection", {}).get("original_n_vertices", 0)
        for r in all_results
    )
    total_filtered = sum(
        r["region"]["n_vertices"]
        for r in all_results
    )
    constrained = sum(
        1 for r in all_results
        if r.get("backprojection", {}).get("constrained_by") is not None
    )

    print(f"Total regions: {len(all_results)}")
    print(f"Regions constrained: {constrained}")
    print(f"Original vertices: {total_original}")
    print(f"Filtered vertices: {total_filtered}")
    if total_original > 0:
        print(f"Retention rate: {total_filtered/total_original:.1%}")


if __name__ == "__main__":
    main()
