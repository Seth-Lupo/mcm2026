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
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    get_contestant_mapping,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Numba-accelerated analytical methods for single elimination
# =============================================================================

@njit(cache=True)
def _is_feasible_numba(ps, pe, lo, hi):
    """Numba-compiled feasibility check."""
    n = len(ps)

    if pe < 1e-12:
        for j in range(n):
            if ps[j] < lo[j] or ps[j] > hi[j]:
                return False
        return True

    # Check simple bounds
    for j in range(n):
        if ps[j] > hi[j]:
            return False
        if ps[j] < lo[j] - pe:
            return False

    # Compute L and U, check intervals and sums
    sum_L = 0.0
    sum_U = 0.0
    for j in range(n):
        L_j = max(0.0, (lo[j] - ps[j]) / pe)
        U_j = min(1.0, (hi[j] - ps[j]) / pe)
        if L_j > U_j + 1e-9:
            return False
        sum_L += L_j
        sum_U += U_j

    return sum_L <= 1.0 + 1e-9 and sum_U >= 1.0 - 1e-9


@njit(cache=True)
def _find_intersection_numba(v_pass, v_fail, elim_idx, surv_idx, lo, hi):
    """
    Numba-compiled edge intersection finder.

    Returns t value where feasibility boundary is crossed.
    """
    n_surv = len(surv_idx)

    # Linear coefficients
    a_surv = np.empty(n_surv)
    b_surv = np.empty(n_surv)
    for i in range(n_surv):
        j = surv_idx[i]
        a_surv[i] = v_pass[j]
        b_surv[i] = v_fail[j] - v_pass[j]

    a_elim = v_pass[elim_idx]
    b_elim = v_fail[elim_idx] - v_pass[elim_idx]

    # Collect candidate t values
    max_candidates = 4 * n_surv + 10
    candidates = np.empty(max_candidates)
    n_cand = 0

    # Simple constraints: p_j <= hi_j
    for i in range(n_surv):
        if abs(b_surv[i]) > 1e-15:
            t = (hi[i] - a_surv[i]) / b_surv[i]
            if 0 < t <= 1:
                candidates[n_cand] = t
                n_cand += 1

    # Simple constraints: p_j + p_elim >= lo_j
    for i in range(n_surv):
        denom = b_surv[i] + b_elim
        if abs(denom) > 1e-15:
            t = (lo[i] - a_surv[i] - a_elim) / denom
            if 0 < t <= 1:
                candidates[n_cand] = t
                n_cand += 1

    # Sort candidates
    candidates_sorted = np.sort(candidates[:n_cand])

    # Check simple constraint crossings first
    for k in range(len(candidates_sorted)):
        t = candidates_sorted[k]
        ps_before = a_surv + b_surv * (t - 1e-9)
        pe_before = a_elim + b_elim * (t - 1e-9)
        ps_after = a_surv + b_surv * (t + 1e-9)
        pe_after = a_elim + b_elim * (t + 1e-9)

        if _is_feasible_numba(ps_before, pe_before, lo, hi) and not _is_feasible_numba(ps_after, pe_after, lo, hi):
            return t

    # Compute breakpoints for piecewise sum constraints
    max_bp = 4 * n_surv + 2
    breakpoints = np.empty(max_bp)
    breakpoints[0] = 0.0
    breakpoints[1] = 1.0
    n_bp = 2

    for i in range(n_surv):
        if abs(b_surv[i]) > 1e-15:
            t = (lo[i] - a_surv[i]) / b_surv[i]
            if 0 < t < 1:
                breakpoints[n_bp] = t
                n_bp += 1
            t = (hi[i] - a_surv[i]) / b_surv[i]
            if 0 < t < 1:
                breakpoints[n_bp] = t
                n_bp += 1

        denom = b_surv[i] + b_elim
        if abs(denom) > 1e-15:
            t = (lo[i] - a_surv[i] - a_elim) / denom
            if 0 < t < 1:
                breakpoints[n_bp] = t
                n_bp += 1
            t = (hi[i] - a_surv[i] - a_elim) / denom
            if 0 < t < 1:
                breakpoints[n_bp] = t
                n_bp += 1

    breakpoints_sorted = np.sort(breakpoints[:n_bp])

    # Check sum constraints in each segment
    sum_candidates = np.empty(2 * n_bp)
    n_sum_cand = 0

    for seg in range(len(breakpoints_sorted) - 1):
        t_lo_seg = breakpoints_sorted[seg]
        t_hi_seg = breakpoints_sorted[seg + 1]
        t_mid = (t_lo_seg + t_hi_seg) / 2

        pe_mid = a_elim + b_elim * t_mid
        if pe_mid < 1e-12:
            continue

        ps_mid = a_surv + b_surv * t_mid

        # Σ L_j = 1 crossing
        sum_lo_active = 0.0
        sum_a_active = 0.0
        sum_b_active = 0.0
        has_active_L = False

        for i in range(n_surv):
            raw_L = (lo[i] - ps_mid[i]) / pe_mid
            if raw_L > 0:
                has_active_L = True
                sum_lo_active += lo[i]
                sum_a_active += a_surv[i]
                sum_b_active += b_surv[i]

        if has_active_L:
            numer = sum_lo_active - sum_a_active - a_elim
            denom = sum_b_active + b_elim
            if abs(denom) > 1e-15:
                t_cross = numer / denom
                if t_lo_seg < t_cross < t_hi_seg:
                    sum_candidates[n_sum_cand] = t_cross
                    n_sum_cand += 1

        # Σ U_j = 1 crossing
        sum_hi_active = 0.0
        sum_a_active = 0.0
        sum_b_active = 0.0
        n_inactive = 0
        has_active_U = False

        for i in range(n_surv):
            raw_U = (hi[i] - ps_mid[i]) / pe_mid
            if raw_U < 1:
                has_active_U = True
                sum_hi_active += hi[i]
                sum_a_active += a_surv[i]
                sum_b_active += b_surv[i]
            else:
                n_inactive += 1

        if has_active_U:
            coef = 1 - n_inactive
            numer = sum_hi_active - sum_a_active - coef * a_elim
            denom = sum_b_active + coef * b_elim
            if abs(denom) > 1e-15:
                t_cross = numer / denom
                if t_lo_seg < t_cross < t_hi_seg:
                    sum_candidates[n_sum_cand] = t_cross
                    n_sum_cand += 1

    # Check sum constraint crossings
    sum_candidates_sorted = np.sort(sum_candidates[:n_sum_cand])
    for k in range(len(sum_candidates_sorted)):
        t = sum_candidates_sorted[k]
        if 0 < t <= 1:
            ps_before = a_surv + b_surv * (t - 1e-9)
            pe_before = a_elim + b_elim * (t - 1e-9)
            ps_after = a_surv + b_surv * (t + 1e-9)
            pe_after = a_elim + b_elim * (t + 1e-9)

            if _is_feasible_numba(ps_before, pe_before, lo, hi) and not _is_feasible_numba(ps_after, pe_after, lo, hi):
                return t

    # Fallback: binary search
    lo_t = 0.0
    hi_t = 1.0
    for _ in range(50):
        mid = (lo_t + hi_t) / 2
        ps_mid = a_surv + b_surv * mid
        pe_mid = a_elim + b_elim * mid
        if _is_feasible_numba(ps_mid, pe_mid, lo, hi):
            lo_t = mid
        else:
            hi_t = mid

    return lo_t


# =============================================================================
# Analytical methods for single elimination (fast path)
# =============================================================================

def check_point_projects_analytical(
    point: np.ndarray,
    elim_idx: int,
    survivor_indices: List[int],
    target_bounds: np.ndarray,
    padding: float = 0.001,
) -> bool:
    """
    Analytical feasibility check for SINGLE elimination case.

    Much faster than LP - pure numpy arithmetic.
    """
    p_elim = point[elim_idx]
    p_survivors = np.array([point[j] for j in survivor_indices])

    # Get bounds for survivors (indexed by their week N+1 position)
    lo = np.array([target_bounds[j][0] for j in range(len(survivor_indices))]) - padding
    hi = np.array([target_bounds[j][1] for j in range(len(survivor_indices))]) + padding

    if p_elim < 1e-12:
        # No redistribution possible, check direct bounds
        return np.all(p_survivors >= lo) and np.all(p_survivors <= hi)

    # Constraint 1: p_j <= hi_j (upper bound reachable with α_j = 0)
    if not np.all(p_survivors <= hi):
        return False

    # Constraint 2: p_j >= lo_j - p_elim (lower bound reachable with α_j = 1)
    if not np.all(p_survivors >= lo - p_elim):
        return False

    # Constraint 3: Σ L_j <= 1 <= Σ U_j
    L = np.maximum(0, (lo - p_survivors) / p_elim)
    U = np.minimum(1, (hi - p_survivors) / p_elim)

    # Each interval must be valid
    if not np.all(L <= U):
        return False

    # Sum constraint
    if L.sum() > 1 + 1e-9 or U.sum() < 1 - 1e-9:
        return False

    return True


def find_edge_intersection_binary(
    v_pass: np.ndarray,
    v_fail: np.ndarray,
    elim_idx: int,
    survivor_indices: List[int],
    target_bounds: np.ndarray,
    padding: float = 0.001,
) -> np.ndarray:
    """
    Find intersection using binary search with analytical feasibility check.
    """
    lo_t, hi_t = 0.0, 1.0

    for _ in range(50):
        mid = (lo_t + hi_t) / 2
        p_mid = (1 - mid) * v_pass + mid * v_fail

        if check_point_projects_analytical(p_mid, elim_idx, survivor_indices,
                                           target_bounds, padding):
            lo_t = mid
        else:
            hi_t = mid

        if hi_t - lo_t < 1e-15:
            break

    return (1 - lo_t) * v_pass + lo_t * v_fail


def find_edge_intersection_analytical(
    v_pass: np.ndarray,
    v_fail: np.ndarray,
    elim_idx: int,
    survivor_indices: List[int],
    target_bounds: np.ndarray,
    padding: float = 0.001,
) -> np.ndarray:
    """
    Find intersection using Numba-accelerated analytical solver.
    """
    surv_idx = np.array(survivor_indices, dtype=np.int64)
    lo = target_bounds[:len(surv_idx), 0] - padding
    hi = target_bounds[:len(surv_idx), 1] + padding

    t = _find_intersection_numba(v_pass, v_fail, elim_idx, surv_idx, lo, hi)
    return (1 - t) * v_pass + t * v_fail


# =============================================================================
# Core projection logic (LP-based for multiple eliminations)
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


# Global shared data for workers (avoid repeated pickling)
_shared_check_params = None


def _init_worker(eliminated_indices, survivor_indices_n, survivor_indices_n1, target_bounds, tolerance, padding):
    """Initialize shared data in worker process."""
    global _shared_check_params
    _shared_check_params = (eliminated_indices, survivor_indices_n, survivor_indices_n1, target_bounds, tolerance, padding)


def _check_vertex_batch(vertices_batch):
    """Check a batch of vertices. Uses shared params initialized in worker."""
    eliminated_indices, survivor_indices_n, survivor_indices_n1, target_bounds, tolerance, padding = _shared_check_params
    results = []
    for v in vertices_batch:
        results.append(check_point_projects_lp(
            v, eliminated_indices, survivor_indices_n, survivor_indices_n1,
            target_bounds, tolerance=tolerance, padding=padding
        ))
    return results


def _find_intersection_batch(edge_batch):
    """Find intersections for a batch of edges. Uses shared params."""
    eliminated_indices, survivor_indices_n, survivor_indices_n1, target_bounds, tolerance, padding = _shared_check_params

    results = []
    for v_pass, v_fail in edge_batch:
        lo, hi = 0.0, 1.0
        tol, max_iter = 1e-4, 15

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            p_mid = (1 - mid) * v_pass + mid * v_fail

            if check_point_projects_lp(p_mid, eliminated_indices, survivor_indices_n,
                                       survivor_indices_n1, target_bounds,
                                       tolerance=tolerance, padding=padding):
                lo = mid
            else:
                hi = mid

            if hi - lo < tol:
                break

        results.append((1 - lo) * v_pass + lo * v_fail)

    return results


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

    Uses ANALYTICAL methods for single elimination (fast), LP for multiple.

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
    n_elim = len(eliminated_indices)

    log.info(f"  Eliminated: {eliminated}")

    vertices = week_n.vertices
    n_verts = len(vertices)

    # Use analytical method for single elimination (MUCH faster)
    use_analytical = (n_elim == 1)

    if use_analytical:
        elim_idx = eliminated_indices[0]
        # Build target bounds array indexed by survivor position in N+1
        target_bounds_arr = week_n1.dim_bounds

        # Step 1: Check all vertices analytically (vectorized, no parallelization needed)
        log.info(f"  Checking {n_verts} vertices (analytical)...")
        passes = np.array([
            check_point_projects_analytical(v, elim_idx, survivor_indices_n,
                                            target_bounds_arr, padding)
            for v in vertices
        ])
    else:
        # Multiple eliminations - use LP with parallelization
        n_workers = min(multiprocessing.cpu_count(), 8)
        log.info(f"  Checking {n_verts} vertices (LP, {n_elim} eliminated)...")

        if n_verts > 50 and n_workers > 1:
            batch_size = max(10, n_verts // (n_workers * 4))
            vertex_batches = [vertices[i:i + batch_size] for i in range(0, n_verts, batch_size)]

            init_args = (eliminated_indices, survivor_indices_n, survivor_indices_n1,
                         week_n1.dim_bounds, tolerance, padding)

            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker,
                                     initargs=init_args) as executor:
                batch_results = list(executor.map(_check_vertex_batch, vertex_batches))

            passes = np.array([r for batch in batch_results for r in batch])
        else:
            check_fn = lambda p: check_point_projects_lp(
                p, eliminated_indices, survivor_indices_n, survivor_indices_n1,
                week_n1.dim_bounds, tolerance=tolerance, padding=padding,
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
            intersection = find_edge_intersection_analytical(
                v_pass, v_fail, elim_idx, survivor_indices_n,
                target_bounds_arr, padding
            )
            result_points.append(intersection)
    else:
        # LP-based with parallelization for multiple eliminations
        n_workers = min(multiprocessing.cpu_count(), 8)

        if n_crossing > 50 and n_workers > 1:
            batch_size = max(10, n_crossing // (n_workers * 4))
            edge_batches = [edge_data[i:i + batch_size] for i in range(0, n_crossing, batch_size)]

            log.info(f"  Finding {n_crossing} intersections in parallel ({n_workers} workers)...")

            init_args = (eliminated_indices, survivor_indices_n, survivor_indices_n1,
                         week_n1.dim_bounds, tolerance, padding)

            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_init_worker,
                                     initargs=init_args) as executor:
                batch_results = list(executor.map(_find_intersection_batch, edge_batches))

            for batch in batch_results:
                result_points.extend(batch)
        else:
            for v_pass, v_fail in edge_data:
                args = (v_pass, v_fail, eliminated_indices, survivor_indices_n,
                        survivor_indices_n1, week_n1.dim_bounds, tolerance, padding)
                result_points.append(_find_single_intersection(args))

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
    max_hull_pts = cfg.hull.max_points  # from config.yaml
    max_dim_exact = cfg.hull.max_dim_exact_volume  # dimensions to use exact ConvexHull

    # For large point sets, subsample first to find approximate hull
    if n_pts > max_hull_pts:
        log.info(f"  Subsampling {n_pts} -> {max_hull_pts} points for hull...")
        rng = np.random.default_rng(cfg.sampling.seed)

        # Keep all original vertices (passing ones), subsample intersections
        n_original = n_pass
        n_intersections = n_pts - n_original

        # Always keep original vertices
        keep_original = min(n_original, max_hull_pts // 2)
        keep_intersections = max_hull_pts - keep_original

        if n_intersections > keep_intersections:
            # Random subsample of intersection points
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


def _process_season_worker(season: int, shared_data: Dict[str, Any]) -> Tuple[List[Dict], List[str]]:
    """
    Worker function to process a single season.
    Called in parallel for each season.
    Returns (results, extra_log_lines).
    """
    from structures import WeekRegion

    regions_data = shared_data["by_season"][season]
    method = shared_data["method"]
    tolerance = shared_data["tolerance"]
    padding = shared_data["padding"]
    n_samples = shared_data["n_samples"]

    # Reconstruct WeekRegion objects
    regions = [WeekRegion.from_dict(r) for r in regions_data]

    results = backproject_season(
        regions,
        method=method,
        tolerance=tolerance,
        padding=padding,
        n_samples=n_samples,
    )

    return results, []


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
    parser.add_argument("--sequential", action="store_true",
                        help="Process seasons sequentially (disable parallelization)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing output file instead of overwriting (only replaces processed seasons)")
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

            season_results = backproject_season(
                by_season[season],
                method=args.method,
                tolerance=bp_cfg.tolerance,
                padding=bp_cfg.bounds_padding,
                n_samples=bp_cfg.n_redistribution_samples,
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
            "method": args.method,
            "tolerance": bp_cfg.tolerance,
            "padding": bp_cfg.bounds_padding,
            "n_samples": bp_cfg.n_redistribution_samples,
        }

        log.info(f"Using {cfg.parallel.n_workers} workers for parallel season processing")

        season_results = run_parallel_seasons(
            seasons=sorted(by_season.keys()),
            process_fn=_process_season_worker,
            shared_data=shared_data,
            task_name="Backprojecting",
        )

        # Collect results in season order
        for season in sorted(season_results.keys()):
            all_results.extend(season_results[season])

    # Save results
    output_path = Path(args.output)
    if args.merge and output_path.exists():
        from structures import merge_results
        seasons_processed = set(by_season.keys())
        merged = merge_results(all_results, output_path, seasons_processed)
        log.info(f"\nMerging {len(all_results)} new regions with existing file ({len(merged)} total)")
        save_results(merged, output_path)
    else:
        log.info(f"\nSaving {len(all_results)} regions to {args.output}")
        save_results(all_results, output_path)

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
