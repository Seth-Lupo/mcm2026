#!/usr/bin/env python3
"""
Forward Projection: Constrain later weeks by earlier weeks.

Under fixed voter preferences, week N's vote distribution constrains
what week N+1 could have been. When contestant X is eliminated, X's voters
redistribute to their next preferred contestant still in the race.

This script filters week N+1's valid region to only points that could have
come from week N's region.

Mirrors backprojection.py but works in the opposite direction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import logging
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from typing import List, Dict, Any, Optional, Tuple
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
# Data structures (same as backprojection)
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
    vertices: Optional[np.ndarray]
    dim_bounds: Optional[np.ndarray]
    centroid: Optional[np.ndarray]
    raw_data: Dict[str, Any]

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
# Volume computation (same as backprojection)
# =============================================================================

def compute_volume(vertices: np.ndarray) -> float:
    """Compute convex hull volume with fallback to approximation."""
    cfg = get_config()

    if len(vertices) < 2:
        return 0.0

    points = vertices[:, :-1]
    n_points, dim = points.shape

    if dim == 0 or n_points < dim + 1:
        return 0.0

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


# =============================================================================
# Core forward projection logic
# =============================================================================

def find_eliminated(week_n: WeekRegion, week_n1: WeekRegion) -> List[str]:
    """Find contestants eliminated between week N and week N+1."""
    return [c for c in week_n.contestants if c not in week_n1.contestants]


def get_contestant_indices(
    week_n: WeekRegion,
    week_n1: WeekRegion,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Get index mappings between weeks.

    Returns:
        eliminated_indices_n: indices of eliminated in week N
        survivor_indices_n: indices of survivors in week N
        survivor_indices_n1: corresponding indices in week N+1
    """
    eliminated = find_eliminated(week_n, week_n1)
    eliminated_indices_n = [week_n.contestants.index(e) for e in eliminated]

    survivor_indices_n = []
    survivor_indices_n1 = []

    for i, name in enumerate(week_n.contestants):
        if name in week_n1.contestants:
            survivor_indices_n.append(i)
            survivor_indices_n1.append(week_n1.contestants.index(name))

    return eliminated_indices_n, survivor_indices_n, survivor_indices_n1


def check_point_came_from_lp(
    point_n1: np.ndarray,
    eliminated_indices_n: List[int],
    survivor_indices_n: List[int],
    survivor_indices_n1: List[int],
    source_bounds: np.ndarray,
    padding: float = 0.001,
) -> bool:
    """
    Check if point in week N+1 could have come from week N's region via LP.

    For point p' in week N+1, we check if there exists:
    - x_j >= 0 for each survivor j (redistributed votes from eliminated to j)
    - Such that:
      - p_j = p'_j - x_j is in week N bounds for survivors
      - e = sum(x_j) is in week N bounds for each eliminated
      - x_j <= p'_j (can't redistribute more than they have)

    For multiple eliminated: x_ij = votes from eliminated i to survivor j
    """
    n_elim = len(eliminated_indices_n)
    n_surv = len(survivor_indices_n)

    if n_elim == 0:
        # No elimination, check if survivors match bounds directly
        for j, (idx_n, idx_n1) in enumerate(zip(survivor_indices_n, survivor_indices_n1)):
            lo, hi = source_bounds[idx_n]
            if point_n1[idx_n1] < lo - padding or point_n1[idx_n1] > hi + padding:
                return False
        return True

    # Variables: x_ij for i in [0, n_elim), j in [0, n_surv)
    # x_ij = votes redistributed from eliminated i to survivor j
    # Flattened: x[i * n_surv + j]
    n_vars = n_elim * n_surv

    # Objective: feasibility only
    c = np.zeros(n_vars)

    # Constraints:
    # For each survivor j:
    #   sum_i x_ij <= p'_j (can't take more than they have)
    #   lo_j <= p'_j - sum_i x_ij <= hi_j (week N survivor bounds)
    #
    # For each eliminated i:
    #   lo_i <= sum_j x_ij <= hi_i (week N eliminated bounds)

    A_ub_list = []
    b_ub_list = []

    # Survivor constraints
    for j, (idx_n, idx_n1) in enumerate(zip(survivor_indices_n, survivor_indices_n1)):
        p_val = point_n1[idx_n1]
        lo, hi = source_bounds[idx_n]
        lo, hi = lo - padding, hi + padding

        # Coefficients for sum_i x_ij
        coeffs = np.zeros(n_vars)
        for i in range(n_elim):
            coeffs[i * n_surv + j] = 1.0

        # sum_i x_ij <= p'_j
        A_ub_list.append(coeffs.copy())
        b_ub_list.append(p_val)

        # p'_j - sum_i x_ij <= hi  =>  -sum_i x_ij <= hi - p'_j
        A_ub_list.append(-coeffs.copy())
        b_ub_list.append(hi - p_val)

        # p'_j - sum_i x_ij >= lo  =>  sum_i x_ij <= p'_j - lo
        A_ub_list.append(coeffs.copy())
        b_ub_list.append(p_val - lo)

    # Eliminated constraints
    for i, elim_idx in enumerate(eliminated_indices_n):
        lo, hi = source_bounds[elim_idx]
        lo, hi = lo - padding, hi + padding

        # Coefficients for sum_j x_ij
        coeffs = np.zeros(n_vars)
        for j in range(n_surv):
            coeffs[i * n_surv + j] = 1.0

        # sum_j x_ij <= hi
        A_ub_list.append(coeffs.copy())
        b_ub_list.append(hi)

        # sum_j x_ij >= lo  =>  -sum_j x_ij <= -lo
        A_ub_list.append(-coeffs.copy())
        b_ub_list.append(-lo)

    A_ub = np.array(A_ub_list) if A_ub_list else None
    b_ub = np.array(b_ub_list) if b_ub_list else None

    # Variable bounds: x_ij >= 0
    bounds = [(0, None) for _ in range(n_vars)]

    # Solve LP
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    return result.success


# =============================================================================
# Region filtering
# =============================================================================

def filter_region_by_forward_projection(
    week_n: WeekRegion,
    week_n1: WeekRegion,
    padding: float = 0.001,
) -> Tuple[np.ndarray, int]:
    """
    Filter week N+1's vertices to those that could have come from week N.

    Returns:
        filtered_vertices: vertices that passed the projection test
        n_filtered: count of filtered vertices
    """
    if week_n1.vertices is None or len(week_n1.vertices) == 0:
        return np.array([]), 0

    if week_n.dim_bounds is None:
        log.warning(f"  Week {week_n.week} has no dim_bounds, skipping constraint")
        return week_n1.vertices, len(week_n1.vertices)

    # Get mappings
    eliminated = find_eliminated(week_n, week_n1)
    eliminated_indices_n, survivor_indices_n, survivor_indices_n1 = get_contestant_indices(week_n, week_n1)

    log.info(f"  Eliminated: {eliminated}")
    log.info(f"  Checking {len(week_n1.vertices)} vertices...")

    # Filter vertices
    valid_mask = np.array([
        check_point_came_from_lp(
            v,
            eliminated_indices_n,
            survivor_indices_n,
            survivor_indices_n1,
            week_n.dim_bounds,
            padding=padding,
        )
        for v in week_n1.vertices
    ])
    filtered = week_n1.vertices[valid_mask]

    return filtered, len(filtered)


def recompute_region_stats(
    vertices: np.ndarray,
    original_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """Recompute region statistics from filtered vertices."""
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

    filtered_volume = compute_volume(vertices)
    diameter = compute_diameter(vertices)

    # Compute relative volume (compared to full simplex)
    n_contestants = vertices.shape[1]
    full_simplex_vol = simplex_volume(n_contestants)
    relative_volume = filtered_volume / full_simplex_vol if full_simplex_vol > 0 else 0.0

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
# Main forward projection pipeline
# =============================================================================

def forwardproject_season(
    regions: List[WeekRegion],
    padding: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    Forward project constraints for a single season.

    Works forward from first week, constraining each week by the previous.
    """
    # Sort by week ascending (work forward)
    sorted_regions = sorted(regions, key=lambda r: (r.is_final, r.week))

    results = []
    prev_region: Optional[WeekRegion] = None

    for region in sorted_regions:
        if prev_region is None:
            # First week - no constraint
            log.info(f"  Week {region.week} (first): no backward constraint")
            result = json.loads(json.dumps(region.raw_data))
            result["forwardprojection"] = {"constrained_by": None}
            results.append(result)
        else:
            # Constrain by previous week
            log.info(f"  Week {prev_region.week} -> Week {region.week}")

            original_volume = region.raw_data.get("region", {}).get("volume", 0.0)

            filtered_vertices, n_filtered = filter_region_by_forward_projection(
                prev_region, region, padding=padding
            )

            log.info(f"  Filtered: {n_filtered}/{len(region.vertices) if region.vertices is not None else 0} vertices")

            result, filtered_volume = recompute_region_stats(filtered_vertices, region.raw_data)

            # Compute volume metrics
            volume_lost = original_volume - filtered_volume
            volume_lost_pct = volume_lost / original_volume if original_volume > 0 else 0.0
            iou = filtered_volume / original_volume if original_volume > 0 else 0.0

            cfg = get_config()
            p = cfg.output.precision

            result["forwardprojection"] = {
                "constrained_by": prev_region.week,
                "original_volume": round(float(original_volume), p + 4),
                "filtered_volume": round(float(filtered_volume), p + 4),
                "volume_lost": round(float(volume_lost), p + 4),
                "volume_lost_pct": round(float(volume_lost_pct), p),
                "iou": round(float(iou), p),
            }
            results.append(result)

        # Update prev_region for next iteration
        # Use the FILTERED region for subsequent constraints
        if len(results) > 0:
            prev_data = results[-1]
            prev_region = WeekRegion.from_dict(prev_data)

    return results


def load_regions(path: Path) -> List[WeekRegion]:
    """Load and parse regions from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [WeekRegion.from_dict(d) for d in data]


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    """Save forward projected results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(description="Forward project region constraints")
    parser.add_argument("--input", "-i", default=str(DATA_DIR / "regions-backprojected.json"),
                        help="Input regions JSON path (default: regions-backprojected.json)")
    parser.add_argument("--output", "-o", default=str(DATA_DIR / "regions-forwardprojected.json"),
                        help="Output path")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to process")
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

        season_results = forwardproject_season(
            by_season[season],
            padding=bp_cfg.bounds_padding,
        )
        all_results.extend(season_results)

    # Save results
    log.info(f"\nSaving {len(all_results)} regions to {args.output}")
    save_results(all_results, Path(args.output))

    # Summary
    print("\n" + "=" * 60)
    print("FORWARD PROJECTION SUMMARY")
    print("=" * 60)

    constrained = sum(
        1 for r in all_results
        if r.get("forwardprojection", {}).get("constrained_by") is not None
    )
    total_orig_vol = sum(
        r.get("forwardprojection", {}).get("original_volume", 0) or 0
        for r in all_results
    )
    total_filt_vol = sum(
        r.get("forwardprojection", {}).get("filtered_volume", 0) or 0
        for r in all_results
    )

    print(f"Total regions: {len(all_results)}")
    print(f"Regions constrained: {constrained}")
    if total_orig_vol > 0:
        print(f"Overall volume retention: {total_filt_vol/total_orig_vol:.1%}")


if __name__ == "__main__":
    main()
