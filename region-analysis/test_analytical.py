#!/usr/bin/env python3
"""
Test analytical feasibility check against LP for single elimination.
"""
import numpy as np
from scipy.optimize import linprog


def check_feasibility_lp(p_survivors, p_elim, lo, hi, padding=1e-9):
    """Original LP-based check."""
    n_surv = len(p_survivors)

    if p_elim < 1e-12:
        # No redistribution possible, check direct bounds
        return np.all(p_survivors >= lo - padding) and np.all(p_survivors <= hi + padding)

    # Variables: α_j for j = 0..n_surv-1
    n_vars = n_surv
    c = np.zeros(n_vars)

    # Equality: Σα_j = 1
    A_eq = np.ones((1, n_vars))
    b_eq = np.array([1.0])

    # Inequality: lo_j <= p_j + α_j * p_elim <= hi_j
    A_ub = []
    b_ub = []

    for j in range(n_surv):
        # α_j * p_elim <= hi_j - p_j  →  α_j <= (hi_j - p_j) / p_elim
        coeffs = np.zeros(n_vars)
        coeffs[j] = p_elim
        A_ub.append(coeffs)
        b_ub.append(hi[j] + padding - p_survivors[j])

        # α_j * p_elim >= lo_j - p_j  →  -α_j <= -(lo_j - p_j) / p_elim
        A_ub.append(-coeffs)
        b_ub.append(-(lo[j] - padding - p_survivors[j]))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(0, 1) for _ in range(n_vars)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    return result.success


def check_feasibility_analytical(p_survivors, p_elim, lo, hi, padding=1e-9):
    """Analytical check for single elimination."""
    n_surv = len(p_survivors)

    lo = lo - padding
    hi = hi + padding

    if p_elim < 1e-12:
        # No redistribution, check direct bounds
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
    if L.sum() > 1 or U.sum() < 1:
        return False

    return True


def test_random_points(n_tests=10000):
    """Test analytical vs LP on random points."""
    np.random.seed(42)

    mismatches = 0

    for _ in range(n_tests):
        # Random number of survivors (2-6)
        n_surv = np.random.randint(2, 7)

        # Random point on simplex (n_surv + 1 contestants)
        point = np.random.dirichlet(np.ones(n_surv + 1))
        p_survivors = point[:n_surv]
        p_elim = point[n_surv]

        # Random bounds (ensuring lo <= hi)
        lo = np.random.uniform(0, 0.5, n_surv)
        hi = lo + np.random.uniform(0.1, 0.5, n_surv)
        hi = np.minimum(hi, 1.0)

        lp_result = check_feasibility_lp(p_survivors, p_elim, lo, hi)
        analytical_result = check_feasibility_analytical(p_survivors, p_elim, lo, hi)

        if lp_result != analytical_result:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH {mismatches}:")
                print(f"  p_survivors: {p_survivors}")
                print(f"  p_elim: {p_elim}")
                print(f"  lo: {lo}")
                print(f"  hi: {hi}")
                print(f"  LP: {lp_result}, Analytical: {analytical_result}")

                # Debug info
                L = np.maximum(0, (lo - p_survivors) / p_elim)
                U = np.minimum(1, (hi - p_survivors) / p_elim)
                print(f"  L: {L}, sum={L.sum()}")
                print(f"  U: {U}, sum={U.sum()}")
                print()

    print(f"Total tests: {n_tests}")
    print(f"Mismatches: {mismatches} ({100*mismatches/n_tests:.2f}%)")
    return mismatches == 0


def find_edge_intersection_analytical(v_pass, v_fail, elim_idx, survivor_indices, lo, hi, padding=1e-9):
    """
    Find intersection of edge with feasibility boundary analytically.

    Edge: p(t) = (1-t)*v_pass + t*v_fail for t in [0, 1]
    v_pass is feasible (t=0), v_fail is infeasible (t=1)

    Returns the point on the boundary (largest t where still feasible).
    """
    lo = lo - padding
    hi = hi + padding
    n_surv = len(survivor_indices)

    # Extract survivor and eliminated components
    def get_point(t):
        return (1 - t) * v_pass + t * v_fail

    def p_surv(t):
        return np.array([(1-t)*v_pass[j] + t*v_fail[j] for j in survivor_indices])

    def p_elim(t):
        return (1-t)*v_pass[elim_idx] + t*v_fail[elim_idx]

    # Linear coefficients: p_j(t) = a_j + b_j * t, p_elim(t) = a_e + b_e * t
    a_surv = np.array([v_pass[j] for j in survivor_indices])
    b_surv = np.array([v_fail[j] - v_pass[j] for j in survivor_indices])
    a_elim = v_pass[elim_idx]
    b_elim = v_fail[elim_idx] - v_pass[elim_idx]

    # Find all candidate t values where constraints become tight
    candidates = []

    # Constraint: p_j <= hi_j
    # a_j + b_j*t = hi_j  →  t = (hi_j - a_j) / b_j
    for j in range(n_surv):
        if abs(b_surv[j]) > 1e-15:
            t_cross = (hi[j] - a_surv[j]) / b_surv[j]
            if 0 < t_cross <= 1:
                candidates.append(t_cross)

    # Constraint: p_j >= lo_j - p_elim  →  p_j + p_elim >= lo_j
    # (a_j + b_j*t) + (a_e + b_e*t) = lo_j
    # t = (lo_j - a_j - a_e) / (b_j + b_e)
    for j in range(n_surv):
        denom = b_surv[j] + b_elim
        if abs(denom) > 1e-15:
            t_cross = (lo[j] - a_surv[j] - a_elim) / denom
            if 0 < t_cross <= 1:
                candidates.append(t_cross)

    # Constraint: Σ L_j <= 1 and Σ U_j >= 1
    # These are piecewise - find breakpoints and check each segment

    # Breakpoints where (lo_j - p_j) / p_elim = 0 or 1
    breakpoints = [0.0, 1.0]
    for j in range(n_surv):
        # (lo_j - p_j(t)) = 0  →  t = (lo_j - a_j) / b_j
        if abs(b_surv[j]) > 1e-15:
            t_bp = (lo[j] - a_surv[j]) / b_surv[j]
            if 0 < t_bp < 1:
                breakpoints.append(t_bp)

        # (lo_j - p_j(t)) / p_elim(t) = 1  →  lo_j - p_j = p_elim
        # lo_j - a_j - b_j*t = a_e + b_e*t  →  t = (lo_j - a_j - a_e) / (b_j + b_e)
        denom = b_surv[j] + b_elim
        if abs(denom) > 1e-15:
            t_bp = (lo[j] - a_surv[j] - a_elim) / denom
            if 0 < t_bp < 1:
                breakpoints.append(t_bp)

        # (hi_j - p_j(t)) = 0  →  t = (hi_j - a_j) / b_j
        if abs(b_surv[j]) > 1e-15:
            t_bp = (hi[j] - a_surv[j]) / b_surv[j]
            if 0 < t_bp < 1:
                breakpoints.append(t_bp)

        # (hi_j - p_j(t)) / p_elim(t) = 1  →  hi_j - p_j = p_elim
        denom = b_surv[j] + b_elim
        if abs(denom) > 1e-15:
            t_bp = (hi[j] - a_surv[j] - a_elim) / denom
            if 0 < t_bp < 1:
                breakpoints.append(t_bp)

    breakpoints = sorted(set(breakpoints))

    # In each segment, find where Σ L_j = 1 or Σ U_j = 1
    for i in range(len(breakpoints) - 1):
        t_lo, t_hi = breakpoints[i], breakpoints[i+1]
        t_mid = (t_lo + t_hi) / 2

        p_e_mid = p_elim(t_mid)
        if p_e_mid < 1e-12:
            continue

        p_s_mid = p_surv(t_mid)

        # Determine which branches of max/min are active at t_mid
        raw_L = (lo - p_s_mid) / p_e_mid
        raw_U = (hi - p_s_mid) / p_e_mid

        # For Σ L_j = 1: sum of max(0, (lo_j - p_j)/p_e) = 1
        # Active terms are where raw_L > 0
        active_L = raw_L > 0

        if active_L.any():
            # Σ_{active} (lo_j - p_j(t)) / p_elim(t) = 1
            # Σ (lo_j - a_j - b_j*t) = a_e + b_e*t
            # Σ lo_j - Σ a_j - (Σ b_j)*t = a_e + b_e*t
            # Σ lo_j - Σ a_j - a_e = (Σ b_j + b_e)*t
            sum_lo = lo[active_L].sum()
            sum_a = a_surv[active_L].sum()
            sum_b = b_surv[active_L].sum()

            numer = sum_lo - sum_a - a_elim
            denom = sum_b + b_elim

            if abs(denom) > 1e-15:
                t_cross = numer / denom
                if t_lo < t_cross < t_hi:
                    candidates.append(t_cross)

        # For Σ U_j = 1: sum of min(1, (hi_j - p_j)/p_e) = 1
        # Inactive terms (U_j = 1) are where raw_U >= 1
        active_U = raw_U < 1
        n_inactive_U = (~active_U).sum()

        if active_U.any():
            # Σ_{active} (hi_j - p_j(t)) / p_elim(t) + n_inactive = 1
            # Σ (hi_j - p_j(t)) = (1 - n_inactive) * p_elim(t)
            sum_hi = hi[active_U].sum()
            sum_a = a_surv[active_U].sum()
            sum_b = b_surv[active_U].sum()

            # sum_hi - sum_a - sum_b*t = (1 - n_inactive) * (a_e + b_e*t)
            # sum_hi - sum_a - sum_b*t = (1-n)*a_e + (1-n)*b_e*t
            # sum_hi - sum_a - (1-n)*a_e = sum_b*t + (1-n)*b_e*t
            coef = 1 - n_inactive_U
            numer = sum_hi - sum_a - coef * a_elim
            denom = sum_b + coef * b_elim

            if abs(denom) > 1e-15:
                t_cross = numer / denom
                if t_lo < t_cross < t_hi:
                    candidates.append(t_cross)

    # Find the smallest t where we transition from feasible to infeasible
    # (i.e., the largest t that is still feasible)
    valid_crossings = []
    for t in candidates:
        if 0 < t <= 1:
            # Check if this is actually on the boundary
            ps = p_surv(t)
            pe = p_elim(t)
            if check_feasibility_analytical(ps, pe, lo + padding, hi + padding, padding=0):
                valid_crossings.append(t)

    if not valid_crossings:
        # No crossing found, use binary search as fallback
        lo_t, hi_t = 0.0, 1.0
        for _ in range(50):
            mid = (lo_t + hi_t) / 2
            ps = p_surv(mid)
            pe = p_elim(mid)
            if check_feasibility_analytical(ps, pe, lo + padding, hi + padding, padding=0):
                lo_t = mid
            else:
                hi_t = mid
        return get_point(lo_t)

    # Return the point at the smallest crossing (boundary)
    t_boundary = min(valid_crossings)
    return get_point(t_boundary)


def find_edge_intersection_binary(v_pass, v_fail, elim_idx, survivor_indices, lo, hi, padding=1e-9):
    """Binary search fallback for comparison."""
    def get_point(t):
        return (1 - t) * v_pass + t * v_fail

    def p_surv(t):
        return np.array([(1-t)*v_pass[j] + t*v_fail[j] for j in survivor_indices])

    def p_elim(t):
        return (1-t)*v_pass[elim_idx] + t*v_fail[elim_idx]

    lo_t, hi_t = 0.0, 1.0
    for _ in range(50):
        mid = (lo_t + hi_t) / 2
        ps = p_surv(mid)
        pe = p_elim(mid)
        if check_feasibility_analytical(ps, pe, lo, hi, padding=padding):
            lo_t = mid
        else:
            hi_t = mid

    return get_point(lo_t)


def test_edge_intersection(n_tests=1000):
    """Test analytical edge intersection against binary search."""
    np.random.seed(123)

    max_error = 0
    total_error = 0

    for test_idx in range(n_tests):
        n_surv = np.random.randint(2, 6)
        n_total = n_surv + 1

        # Random points on simplex
        v_pass = np.random.dirichlet(np.ones(n_total))
        v_fail = np.random.dirichlet(np.ones(n_total))

        elim_idx = n_total - 1  # Last one is eliminated
        survivor_indices = list(range(n_surv))

        # Random bounds
        lo = np.random.uniform(0, 0.3, n_surv)
        hi = lo + np.random.uniform(0.2, 0.5, n_surv)
        hi = np.minimum(hi, 1.0)

        # Check that v_pass is feasible and v_fail is not
        p_pass = v_pass[survivor_indices]
        e_pass = v_pass[elim_idx]
        p_fail = v_fail[survivor_indices]
        e_fail = v_fail[elim_idx]

        pass_feas = check_feasibility_analytical(p_pass, e_pass, lo, hi)
        fail_feas = check_feasibility_analytical(p_fail, e_fail, lo, hi)

        if not (pass_feas and not fail_feas):
            continue  # Skip if not a valid test case

        # Find intersections
        p_analytical = find_edge_intersection_analytical(v_pass, v_fail, elim_idx, survivor_indices, lo, hi)
        p_binary = find_edge_intersection_binary(v_pass, v_fail, elim_idx, survivor_indices, lo, hi)

        error = np.linalg.norm(p_analytical - p_binary)
        max_error = max(max_error, error)
        total_error += error

        if error > 0.01:
            print(f"Test {test_idx}: Large error {error:.6f}")
            print(f"  Analytical: {p_analytical}")
            print(f"  Binary:     {p_binary}")

    print(f"\nEdge intersection tests: {n_tests}")
    print(f"Max error: {max_error:.2e}")
    print(f"Avg error: {total_error/n_tests:.2e}")

    return max_error < 0.001


if __name__ == "__main__":
    print("=== Testing feasibility check ===")
    success1 = test_random_points(10000)
    if success1:
        print("\n✓ Analytical feasibility check matches LP perfectly!")
    else:
        print("\n✗ There are mismatches in feasibility check")

    print("\n=== Testing edge intersection ===")
    success2 = test_edge_intersection(1000)
    if success2:
        print("\n✓ Analytical edge intersection matches binary search!")
    else:
        print("\n✗ Edge intersection has errors")
