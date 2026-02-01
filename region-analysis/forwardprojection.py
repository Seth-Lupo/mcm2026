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
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from config import get_config
from geometry import simplex_volume, compute_volume, compute_diameter
from structures import (
    WeekRegion,
    load_regions,
    save_results,
    recompute_region_stats,
    find_eliminated,
    get_contestant_indices,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Core forward projection logic
# =============================================================================

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

def get_hull_edges(vertices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get edges of the convex hull.

    For high dimensions, falls back to all pairs (conservative but slower).
    """
    n_verts, dim = vertices.shape

    if dim <= 6 and n_verts >= dim + 1:
        try:
            from scipy.spatial import ConvexHull
            projected = vertices[:, :-1]
            hull = ConvexHull(projected)

            edges = set()
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                        edges.add(edge)
            return list(edges)
        except Exception:
            pass

    return [(i, j) for i in range(n_verts) for j in range(i + 1, n_verts)]


def _find_single_intersection_forward(args):
    """
    Worker function for parallel edge intersection finding.
    """
    v_pass, v_fail, eliminated_indices_n, survivor_indices_n, survivor_indices_n1, source_bounds, padding = args

    def check_fn(p):
        return check_point_came_from_lp(
            p, eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
            source_bounds, padding=padding
        )

    lo, hi = 0.0, 1.0
    tol, max_iter = 1e-4, 15

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


def filter_region_by_forward_projection(
    week_n: WeekRegion,
    week_n1: WeekRegion,
    padding: float = 0.001,
) -> Tuple[np.ndarray, int]:
    """
    Compute the intersection of week N+1's region with the forward projection constraint.

    Uses parallel processing for edge intersection finding.
    """
    cfg = get_config()

    if week_n1.vertices is None or len(week_n1.vertices) == 0:
        return np.array([]), 0

    if week_n.dim_bounds is None:
        log.warning(f"  Week {week_n.week} has no dim_bounds, skipping constraint")
        return week_n1.vertices, len(week_n1.vertices)

    # Get mappings
    eliminated = find_eliminated(week_n, week_n1)
    eliminated_indices_n, survivor_indices_n, survivor_indices_n1 = get_contestant_indices(week_n, week_n1)

    log.info(f"  Eliminated: {eliminated}")

    # Build check function for vertex testing
    check_fn = lambda p: check_point_came_from_lp(
        p, eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
        week_n.dim_bounds, padding=padding,
    )

    vertices = week_n1.vertices
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

    # Sample edges if too many
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
            eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
            week_n.dim_bounds, padding
        ))

    # Parallel edge intersection finding
    n_workers = min(multiprocessing.cpu_count(), 8, n_crossing)

    if n_crossing > 50 and n_workers > 1:
        log.info(f"  Finding intersections in parallel ({n_workers} workers)...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            intersections = list(executor.map(_find_single_intersection_forward, edge_args, chunksize=max(1, n_crossing // n_workers // 4)))
        result_points.extend(intersections)
    else:
        for args in edge_args:
            result_points.append(_find_single_intersection_forward(args))

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
