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
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import get_config
from geometry import simplex_volume, compute_volume, compute_diameter
from structures import (
    WeekRegion,
    load_regions,
    save_results,
    recompute_region_stats,
    find_eliminated,
    get_contestant_mapping,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Core projection logic
# =============================================================================

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
        cfg = get_config()
        rng = np.random.default_rng(cfg.sampling.seed)

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

def get_hull_edges(vertices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get edges of the convex hull.

    For high dimensions, falls back to all pairs (conservative but slower).

    Returns:
        List of (i, j) index pairs representing edges
    """
    n_verts, dim = vertices.shape

    # For low dimensions, use scipy ConvexHull to get actual edges
    if dim <= 6 and n_verts >= dim + 1:
        try:
            from scipy.spatial import ConvexHull
            # Project to n-1 dims for hull computation
            projected = vertices[:, :-1]
            hull = ConvexHull(projected)

            # Extract unique edges from simplices
            edges = set()
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                        edges.add(edge)
            return list(edges)
        except Exception:
            pass

    # Fallback: all pairs (conservative)
    return [(i, j) for i in range(n_verts) for j in range(i + 1, n_verts)]


def _find_single_intersection(args):
    """
    Worker function for parallel edge intersection finding.
    Must be module-level (not nested) to be picklable.
    """
    v_pass, v_fail, eliminated_indices, survivor_indices_n, survivor_indices_n1, target_bounds, tolerance, padding = args

    def check_fn(p):
        return check_point_projects_lp(
            p, eliminated_indices, survivor_indices_n, survivor_indices_n1,
            target_bounds, tolerance=tolerance, padding=padding
        )

    lo, hi = 0.0, 1.0
    tol, max_iter = 1e-4, 15  # Coarse but fast

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p_mid = (1 - mid) * v_pass + mid * v_fail

        if check_fn(p_mid):
            lo = mid
        else:
            hi = mid

        if hi - lo < tol:
            break

    return (1 - lo) * v_pass + lo * v_fail


def filter_region_by_projection(
    week_n: WeekRegion,
    week_n1: WeekRegion,
    method: str = "lp",
    tolerance: float = 1e-9,
    padding: float = 0.001,
    n_samples: int = 1000,
) -> Tuple[np.ndarray, int]:
    """
    Compute the intersection of week N's region with the backprojection constraint.

    When a constraint hyperplane cuts through the polytope, original vertices
    may be removed but NEW vertices appear where edges cross the boundary.

    Uses parallel processing for edge intersection finding.

    Returns:
        filtered_vertices: vertices of the constrained polytope
        n_filtered: count of vertices
    """
    cfg = get_config()

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

    # Build check function for vertex testing
    check_fn = lambda p: check_point_projects_lp(
        p, eliminated_indices, survivor_indices_n, survivor_indices_n1,
        week_n1.dim_bounds, tolerance=tolerance, padding=padding,
    )

    vertices = week_n.vertices
    n_verts = len(vertices)

    # Step 1: Check all vertices
    log.info(f"  Checking {n_verts} vertices...")
    passes = np.array([check_fn(v) for v in vertices])
    n_pass = passes.sum()
    log.info(f"  {n_pass}/{n_verts} vertices pass")

    if n_pass == n_verts:
        return vertices, n_verts

    if n_pass == 0:
        return np.array([]), 0

    # Step 2: Collect passing vertices
    result_points = [vertices[i] for i in range(n_verts) if passes[i]]

    # Step 3: Find edge intersections
    edges = get_hull_edges(vertices)
    crossing_edges = [(i, j) for i, j in edges if passes[i] != passes[j]]
    n_crossing = len(crossing_edges)

    log.info(f"  {n_crossing} edges cross boundary (of {len(edges)} total)")

    if n_crossing == 0:
        all_points = np.array(result_points)
        return all_points, len(all_points)

    # Sample edges if too many (extreme points will still find boundary)
    max_edges = 500
    if n_crossing > max_edges:
        rng = np.random.default_rng(cfg.sampling.seed)
        sampled_indices = rng.choice(n_crossing, max_edges, replace=False)
        crossing_edges = [crossing_edges[i] for i in sampled_indices]
        log.info(f"  Sampled {max_edges} edges (was {n_crossing})")
        n_crossing = max_edges

    # Prepare args for parallel processing
    edge_args = []
    for i, j in crossing_edges:
        if passes[i]:
            v_pass, v_fail = vertices[i], vertices[j]
        else:
            v_pass, v_fail = vertices[j], vertices[i]

        edge_args.append((
            v_pass, v_fail,
            eliminated_indices, survivor_indices_n, survivor_indices_n1,
            week_n1.dim_bounds, tolerance, padding
        ))

    # Parallel edge intersection finding
    n_workers = min(multiprocessing.cpu_count(), 8, n_crossing)

    if n_crossing > 50 and n_workers > 1:
        log.info(f"  Finding intersections in parallel ({n_workers} workers)...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            intersections = list(executor.map(_find_single_intersection, edge_args, chunksize=max(1, n_crossing // n_workers // 4)))
        result_points.extend(intersections)
    else:
        # Sequential for small number of edges
        for args in edge_args:
            result_points.append(_find_single_intersection(args))

    log.info(f"  Added {n_crossing} edge intersection points")

    if len(result_points) == 0:
        return np.array([]), 0

    all_points = np.array(result_points)

    # Step 4: Extract hull vertices
    from geometry import find_extreme_points
    if len(all_points) > 10:
        projected = all_points[:, :-1]
        result = find_extreme_points(projected, progress_fn=None)
        hull_vertices = all_points[result["vertex_indices"]]
        log.info(f"  Final: {len(hull_vertices)} hull vertices")
        return hull_vertices, len(hull_vertices)

    return all_points, len(all_points)


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

    constrained = [
        r for r in all_results
        if r.get("backprojection", {}).get("constrained_by") is not None
    ]
    total_filtered = sum(r["region"]["n_vertices"] for r in all_results)

    # Volume-based metrics (more meaningful than vertex counts)
    total_orig_vol = sum(
        r.get("backprojection", {}).get("original_volume", 0) or 0
        for r in constrained
    )
    total_filt_vol = sum(
        r.get("backprojection", {}).get("filtered_volume", 0) or 0
        for r in constrained
    )

    print(f"Total regions: {len(all_results)}")
    print(f"Regions constrained: {len(constrained)}")
    print(f"Total vertices after filtering: {total_filtered}")
    if total_orig_vol > 0:
        print(f"Overall volume retention: {total_filt_vol/total_orig_vol:.1%}")


if __name__ == "__main__":
    main()
