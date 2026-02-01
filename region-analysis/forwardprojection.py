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
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import njit

from config import get_config
from geometry import simplex_volume, compute_volume, compute_diameter, add_jitter
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
# Numba-accelerated analytical methods for single elimination
# =============================================================================

@njit(cache=True)
def _is_feasible_forward_numba(ps_prime, lo_surv, hi_surv, lo_elim, hi_elim):
    """
    Numba-compiled feasibility check for forward projection.

    Given point p' in week N+1, check if it could have come from week N.
    For single elimination: x_j = votes redistributed from eliminated to survivor j.
    Original survivor j had: p_j = p'_j - x_j, must be in [lo_j, hi_j].
    Eliminated had: e = Σx_j, must be in [lo_e, hi_e].
    """
    n = len(ps_prime)

    sum_L = 0.0
    sum_U = 0.0

    for j in range(n):
        # Bounds on x_j:
        # x_j >= 0
        # x_j <= p'_j (can't take more than they have)
        # x_j >= p'_j - hi_j (from p'_j - x_j <= hi_j)
        # x_j <= p'_j - lo_j (from p'_j - x_j >= lo_j)
        L_j = max(0.0, ps_prime[j] - hi_surv[j])
        U_j = min(ps_prime[j], ps_prime[j] - lo_surv[j])

        if L_j > U_j + 1e-9:
            return False

        sum_L += L_j
        sum_U += U_j

    # Sum constraint: lo_e <= Σx_j <= hi_e
    # Combined with interval constraints: max(sum_L, lo_e) <= min(sum_U, hi_e)
    lower = max(sum_L, lo_elim)
    upper = min(sum_U, hi_elim)

    return lower <= upper + 1e-9


@njit(cache=True)
def _find_intersection_forward_numba(v_pass, v_fail, surv_idx_n1, lo_surv, hi_surv, lo_elim, hi_elim):
    """
    Numba-compiled edge intersection finder for forward projection.
    Returns t value where feasibility boundary is crossed.
    """
    n_surv = len(surv_idx_n1)

    # Linear coefficients: p'(t) = a + b*t where a = v_pass, b = v_fail - v_pass
    a_surv = np.empty(n_surv)
    b_surv = np.empty(n_surv)
    for i in range(n_surv):
        j = surv_idx_n1[i]
        a_surv[i] = v_pass[j]
        b_surv[i] = v_fail[j] - v_pass[j]

    # Collect candidate t values from constraint boundaries
    max_candidates = 4 * n_surv + 10
    candidates = np.empty(max_candidates)
    n_cand = 0

    # Candidates from L_j = U_j crossings and sum constraint crossings
    for i in range(n_surv):
        # L_j = max(0, p'_j - hi_j) changes at p'_j = hi_j
        if abs(b_surv[i]) > 1e-15:
            t = (hi_surv[i] - a_surv[i]) / b_surv[i]
            if 0 < t <= 1:
                candidates[n_cand] = t
                n_cand += 1

        # U_j = min(p'_j, p'_j - lo_j) = p'_j - lo_j (since lo >= 0)
        # Changes behavior at p'_j = lo_j (but p'_j >= lo_j always for valid)
        if abs(b_surv[i]) > 1e-15:
            t = (lo_surv[i] - a_surv[i]) / b_surv[i]
            if 0 < t <= 1:
                candidates[n_cand] = t
                n_cand += 1

    # Sort and check candidates
    candidates_sorted = np.sort(candidates[:n_cand])

    for k in range(len(candidates_sorted)):
        t = candidates_sorted[k]
        ps_before = a_surv + b_surv * (t - 1e-9)
        ps_after = a_surv + b_surv * (t + 1e-9)

        feas_before = _is_feasible_forward_numba(ps_before, lo_surv, hi_surv, lo_elim, hi_elim)
        feas_after = _is_feasible_forward_numba(ps_after, lo_surv, hi_surv, lo_elim, hi_elim)

        if feas_before and not feas_after:
            return t

    # Fallback: binary search
    lo_t = 0.0
    hi_t = 1.0
    for _ in range(50):
        mid = (lo_t + hi_t) / 2
        ps_mid = a_surv + b_surv * mid
        if _is_feasible_forward_numba(ps_mid, lo_surv, hi_surv, lo_elim, hi_elim):
            lo_t = mid
        else:
            hi_t = mid

    return lo_t


def check_point_came_from_analytical(
    point_n1: np.ndarray,
    elim_idx_n: int,
    survivor_indices_n: list,
    survivor_indices_n1: list,
    source_bounds: np.ndarray,
    padding: float = 0.001,
) -> bool:
    """
    Analytical feasibility check for SINGLE elimination forward projection.
    """
    n_surv = len(survivor_indices_n)

    # Get survivor values in week N+1
    ps_prime = np.array([point_n1[j] for j in survivor_indices_n1])

    # Get survivor bounds from week N
    lo_surv = np.array([source_bounds[j][0] for j in survivor_indices_n]) - padding
    hi_surv = np.array([source_bounds[j][1] for j in survivor_indices_n]) + padding

    # Get eliminated bounds from week N
    lo_elim = source_bounds[elim_idx_n][0] - padding
    hi_elim = source_bounds[elim_idx_n][1] + padding

    return _is_feasible_forward_numba(ps_prime, lo_surv, hi_surv, lo_elim, hi_elim)


def find_edge_intersection_forward_analytical(
    v_pass: np.ndarray,
    v_fail: np.ndarray,
    elim_idx_n: int,
    survivor_indices_n: list,
    survivor_indices_n1: list,
    source_bounds: np.ndarray,
    padding: float = 0.001,
) -> np.ndarray:
    """
    Find intersection using Numba-accelerated analytical solver for forward projection.
    """
    surv_idx_n1 = np.array(survivor_indices_n1, dtype=np.int64)

    # Get survivor bounds from week N
    lo_surv = np.array([source_bounds[j][0] for j in survivor_indices_n]) - padding
    hi_surv = np.array([source_bounds[j][1] for j in survivor_indices_n]) + padding

    # Get eliminated bounds from week N
    lo_elim = source_bounds[elim_idx_n][0] - padding
    hi_elim = source_bounds[elim_idx_n][1] + padding

    t = _find_intersection_forward_numba(v_pass, v_fail, surv_idx_n1, lo_surv, hi_surv, lo_elim, hi_elim)
    return (1 - t) * v_pass + t * v_fail


# =============================================================================
# Core forward projection logic (LP-based for multiple eliminations)
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


# Global shared data for workers (avoid repeated pickling)
_shared_check_params_fwd = None


def _init_worker_forward(eliminated_indices_n, survivor_indices_n, survivor_indices_n1, source_bounds, padding):
    """Initialize shared data in worker process."""
    global _shared_check_params_fwd
    _shared_check_params_fwd = (eliminated_indices_n, survivor_indices_n, survivor_indices_n1, source_bounds, padding)


def _check_vertex_batch_forward(vertices_batch):
    """Check a batch of vertices. Uses shared params initialized in worker."""
    eliminated_indices_n, survivor_indices_n, survivor_indices_n1, source_bounds, padding = _shared_check_params_fwd
    results = []
    for v in vertices_batch:
        results.append(check_point_came_from_lp(
            v, eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
            source_bounds, padding=padding
        ))
    return results


def _find_intersection_batch_forward(edge_batch):
    """Find intersections for a batch of edges. Uses shared params."""
    eliminated_indices_n, survivor_indices_n, survivor_indices_n1, source_bounds, padding = _shared_check_params_fwd

    results = []
    for v_pass, v_fail in edge_batch:
        lo, hi = 0.0, 1.0
        tol, max_iter = 1e-4, 15

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            p_mid = (1 - mid) * v_pass + mid * v_fail

            if check_point_came_from_lp(p_mid, eliminated_indices_n, survivor_indices_n,
                                        survivor_indices_n1, source_bounds, padding=padding):
                lo = mid
            else:
                hi = mid

            if hi - lo < tol:
                break

        results.append((1 - lo) * v_pass + lo * v_fail)

    return results


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

    Uses ANALYTICAL methods for single elimination (fast), LP for multiple.
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
    n_elim = len(eliminated_indices_n)

    log.info(f"  Eliminated: {eliminated}")

    vertices = week_n1.vertices
    n_verts = len(vertices)

    # Use analytical method for single elimination (MUCH faster)
    use_analytical = (n_elim == 1)

    if use_analytical:
        elim_idx_n = eliminated_indices_n[0]

        # Step 1: Check all vertices analytically
        log.info(f"  Checking {n_verts} vertices (analytical)...")
        passes = np.array([
            check_point_came_from_analytical(v, elim_idx_n, survivor_indices_n,
                                             survivor_indices_n1, week_n.dim_bounds, padding)
            for v in vertices
        ])
    else:
        # Multiple eliminations - use LP with parallelization
        n_workers = min(multiprocessing.cpu_count(), 8)
        log.info(f"  Checking {n_verts} vertices (LP, {n_elim} eliminated)...")

        if n_verts > 50 and n_workers > 1:
            batch_size = max(10, n_verts // (n_workers * 4))
            vertex_batches = [vertices[i:i + batch_size] for i in range(0, n_verts, batch_size)]

            init_args = (eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
                         week_n.dim_bounds, padding)

            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker_forward,
                                     initargs=init_args) as executor:
                batch_results = list(executor.map(_check_vertex_batch_forward, vertex_batches))

            passes = np.array([r for batch in batch_results for r in batch])
        else:
            check_fn = lambda p: check_point_came_from_lp(
                p, eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
                week_n.dim_bounds, padding=padding,
            )
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

    # Prepare edge data
    edge_data = []
    for i, j in crossing_edges:
        if passes[i]:
            edge_data.append((vertices[i], vertices[j]))
        else:
            edge_data.append((vertices[j], vertices[i]))

    if use_analytical:
        # Analytical edge intersection (no parallelization needed - it's fast)
        log.info(f"  Finding {n_crossing} intersections (analytical)...")
        for v_pass, v_fail in edge_data:
            intersection = find_edge_intersection_forward_analytical(
                v_pass, v_fail, elim_idx_n, survivor_indices_n,
                survivor_indices_n1, week_n.dim_bounds, padding
            )
            result_points.append(intersection)
    else:
        # LP-based with parallelization for multiple eliminations
        n_workers = min(multiprocessing.cpu_count(), 8)

        if n_crossing > 50 and n_workers > 1:
            batch_size = max(10, n_crossing // (n_workers * 4))
            edge_batches = [edge_data[i:i + batch_size] for i in range(0, n_crossing, batch_size)]

            log.info(f"  Finding {n_crossing} intersections in parallel ({n_workers} workers)...")

            init_args = (eliminated_indices_n, survivor_indices_n, survivor_indices_n1,
                         week_n.dim_bounds, padding)

            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker_forward,
                                     initargs=init_args) as executor:
                batch_results = list(executor.map(_find_intersection_batch_forward, edge_batches))

            for batch in batch_results:
                result_points.extend(batch)
        else:
            for v_pass, v_fail in edge_data:
                args = (v_pass, v_fail, eliminated_indices_n, survivor_indices_n,
                        survivor_indices_n1, week_n.dim_bounds, padding)
                result_points.append(_find_single_intersection_forward(args))

    log.info(f"  Added {n_crossing} edge intersection points")

    if len(result_points) == 0:
        return np.array([]), 0

    all_points = np.array(result_points)

    # Step 4: Extract hull vertices using config settings
    if len(all_points) <= 10:
        return all_points, len(all_points)

    from scipy.spatial import ConvexHull
    projected = all_points[:, :-1]  # Project to n-1 dims (simplex constraint)
    n_pts, dim = projected.shape

    # Use config settings for hull extraction
    max_hull_pts = cfg.hull.max_points
    max_dim_exact = cfg.hull.max_dim_exact_volume

    # For large point sets, subsample first
    if n_pts > max_hull_pts:
        log.info(f"  Subsampling {n_pts} -> {max_hull_pts} points for hull...")
        rng = np.random.default_rng(cfg.sampling.seed)

        # Keep all original vertices (passing ones), subsample intersections
        n_original = n_pass
        n_intersections = n_pts - n_original

        keep_original = min(n_original, max_hull_pts // 2)
        keep_intersections = max_hull_pts - keep_original

        if n_intersections > keep_intersections:
            intersection_indices = np.arange(n_original, n_pts)
            sampled_intersection_indices = rng.choice(
                intersection_indices, keep_intersections, replace=False
            )
            keep_indices = np.concatenate([
                np.arange(keep_original),
                sampled_intersection_indices
            ])
        else:
            keep_indices = np.arange(n_pts)

        projected_sub = projected[keep_indices]
        all_points_sub = all_points[keep_indices]
    else:
        projected_sub = projected
        all_points_sub = all_points

    # Try exact ConvexHull for lower dimensions
    if dim <= max_dim_exact and len(projected_sub) > dim + 1:
        try:
            hull = ConvexHull(projected_sub)
            hull_indices = np.unique(hull.simplices.flatten())
            hull_vertices = all_points_sub[hull_indices]
            log.info(f"  Final: {len(hull_vertices)} hull vertices (ConvexHull)")
            return hull_vertices, len(hull_vertices)
        except Exception as e:
            # Try with jitter for degenerate cases
            err_line = str(e).split('\n')[0][:80]
            log.warning(f"  ConvexHull failed ({err_line}...), retrying with jitter...")
            try:
                jittered = add_jitter(projected_sub)
                hull = ConvexHull(jittered)
                hull_indices = np.unique(hull.simplices.flatten())
                hull_vertices = all_points_sub[hull_indices]
                log.info(f"  Final: {len(hull_vertices)} hull vertices (ConvexHull with jitter)")
                return hull_vertices, len(hull_vertices)
            except Exception as e2:
                err_line2 = str(e2).split('\n')[0][:80]
                log.warning(f"  ConvexHull with jitter also failed ({err_line2}...), using extreme points")

    # Fallback: random directions (uses cfg.hull.n_directions internally)
    from geometry import find_extreme_points
    result = find_extreme_points(projected_sub, progress_fn=None)
    hull_vertices = all_points_sub[result["vertex_indices"]]
    log.info(f"  Final: {len(hull_vertices)} hull vertices")
    return hull_vertices, len(hull_vertices)


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


def _process_season_worker(season: int, shared_data: Dict[str, Any]) -> Tuple[List[Dict], List[str]]:
    """
    Worker function to process a single season for forward projection.
    Called in parallel for each season.
    Returns (results, extra_log_lines).
    """
    from structures import WeekRegion

    regions_data = shared_data["by_season"][season]
    padding = shared_data["padding"]

    # Reconstruct WeekRegion objects
    regions = [WeekRegion.from_dict(r) for r in regions_data]

    results = forwardproject_season(
        regions,
        padding=padding,
    )

    return results, []


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser(description="Forward project region constraints")
    parser.add_argument("--input", "-i", default=str(DATA_DIR / "regions-backprojected.json"),
                        help="Input regions JSON path (default: regions-backprojected.json)")
    parser.add_argument("--output", "-o", default=str(DATA_DIR / "regions-forwardprojected.json"),
                        help="Output path")
    parser.add_argument("--seasons", "-s", default=None,
                        help="Comma-separated seasons to process")
    parser.add_argument("--sequential", action="store_true",
                        help="Process seasons sequentially (disable parallelization)")
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

    n_seasons = len(by_season)
    log.info(f"Processing {n_seasons} seasons")

    bp_cfg = cfg.backprojection
    all_results = []

    if args.sequential or n_seasons <= 1:
        # Sequential processing
        for season in sorted(by_season.keys()):
            log.info(f"\n{'='*60}")
            log.info(f"Season {season}")
            log.info(f"{'='*60}")

            season_results = forwardproject_season(
                by_season[season],
                padding=bp_cfg.bounds_padding,
            )
            all_results.extend(season_results)
    else:
        # Parallel processing of seasons
        from parallel import run_parallel_seasons

        # Serialize region data for workers
        by_season_serialized = {
            s: [r.raw_data for r in rs]
            for s, rs in by_season.items()
        }

        shared_data = {
            "by_season": by_season_serialized,
            "padding": bp_cfg.bounds_padding,
        }

        log.info(f"Using {cfg.parallel.n_workers} workers for parallel season processing")

        season_results = run_parallel_seasons(
            seasons=sorted(by_season.keys()),
            process_fn=_process_season_worker,
            shared_data=shared_data,
            task_name="Forward projecting",
        )

        # Collect results in season order
        for season in sorted(season_results.keys()):
            all_results.extend(season_results[season])

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
