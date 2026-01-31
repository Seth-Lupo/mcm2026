# --------- INPUTS ----------
X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]          # length n, usually a permutation of 1..n
P = [1,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0]e          # one-hot; P[i_star] = 1

# Rank vectors to check membership (leave empty if none)
CHECK_VECTORS = [
    list(range(15,0,-1)),
    list(range(1,16, 1))
]
# ---------------------------

from collections import defaultdict

def i_star_from_P(P):
    ones = [i for i,v in enumerate(P) if v == 1]
    if len(ones) != 1:
        raise ValueError("P must be one-hot (exactly one 1).")
    return ones[0]

def is_valid_solution(X, i_star, Y):
    """Check if Y is a valid solution (i_star wins with its sum)."""
    n = len(X)

    # Check Y is a valid permutation of 1..n
    if len(Y) != n:
        return False
    if sorted(Y) != list(range(1, n+1)):
        return False

    # Compute sums
    sums = [X[i] + Y[i] for i in range(n)]
    s_star = sums[i_star]
    k = Y[i_star]

    # Check i_star wins: its sum must be <= all others,
    # and for ties, its Y value (k) must be smallest
    for j in range(n):
        if j == i_star:
            continue
        if sums[j] < s_star:
            return False
        if sums[j] == s_star and Y[j] < k:
            return False
    return True

def thresholds_for_k(X, i_star, k):
    n = len(X)
    s = X[i_star] + k
    thr = [None]*n
    thr[i_star] = k
    for j in range(n):
        if j == i_star:
            continue
        L = s - X[j]                    # need y_j >= L to avoid smaller sum
        low = max(1, L)
        if 1 <= L <= n and L < k:       # tie (sum==s) would beat k, so forbid y_j=L
            low = max(low, L + 1)
        thr[j] = low
        if low > n:
            return None
    return thr

def allowed_mask(n, k, low):
    m = 0
    for v in range(low, n+1):
        if v != k:
            m |= 1 << (v-1)
    return m

def count_for_k(X, i_star, k):
    n = len(X)
    thr = thresholds_for_k(X, i_star, k)
    if thr is None:
        return 0

    positions = [j for j in range(n) if j != i_star]
    masks = []
    for j in positions:
        masks.append(allowed_mask(n, k, thr[j]))
        if masks[-1] == 0:
            return 0

    # assign hardest-first for faster DP
    order = sorted(range(len(positions)), key=lambda t: (masks[t].bit_count(), -thr[positions[t]]))
    positions = [positions[t] for t in order]
    masks = [masks[t] for t in order]

    start_mask = 1 << (k-1)
    dp = {start_mask: 1}
    for m_allowed in masks:
        ndp = defaultdict(int)
        for used, cnt in dp.items():
            avail = m_allowed & ~used
            a = avail
            while a:
                lsb = a & -a
                ndp[used | lsb] += cnt
                a -= lsb
        dp = ndp
        if not dp:
            return 0
    return sum(dp.values())

def main():
    n = len(X)
    if len(P) != n:
        raise ValueError("X and P must have same length.")
    i_star = i_star_from_P(P)

    total = 0
    feasible = []
    per_k = {}

    for k in range(1, n+1):
        c = count_for_k(X, i_star, k)
        if c > 0:
            feasible.append(k)
            per_k[k] = c
            total += c

    print(f"n={n}")
    print(f"X={X}")
    print(f"i_star={i_star} (0-based), P has 1 at this index")
    print(f"feasible k=y[i_star]: {feasible}")
    print("counts by k:", per_k)
    print("TOTAL #Y:", total)

    # Check membership of specified vectors
    if CHECK_VECTORS:
        print("\n--- Membership checks ---")
        for Y in CHECK_VECTORS:
            valid = is_valid_solution(X, i_star, Y)
            status = "IN SET" if valid else "NOT IN SET"
            print(f"{Y} -> {status}")

if __name__ == "__main__":
    main()
