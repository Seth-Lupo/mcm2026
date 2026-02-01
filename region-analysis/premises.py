"""
Centralized premise logic for DWTS vote analysis.

This is the SINGLE SOURCE OF TRUTH for all premise validation.
All other modules should import from here.

Premise Types:
- RR (Rank-Rank): Seasons 1-2, 28-34. Convert scores to ranks, sum ranks.
- PP (Percent-Percent): Seasons 3-27. Convert scores to percentages, sum percentages.
- B2 (Bottom-2): Judge saves one from bottom 2 (some middle seasons).
- FINAL: Final episode ranking (all contestants get placements).
"""
import numpy as np
from enum import Enum, auto
from typing import List, Set, Optional
from dataclasses import dataclass
from numba import njit, prange


# =============================================================================
# PREMISE TYPES
# =============================================================================

class PremiseType(Enum):
    """Premise types for DWTS elimination/ranking."""
    RR_ELIM = auto()       # Single elimination, rank-rank
    RR_MULTI = auto()      # Multiple elimination, rank-rank
    RR_B2 = auto()         # Bottom-2 with judge save, rank-rank
    RR_B2_MULTI = auto()   # Bottom-2 multiple, rank-rank
    RR_FINAL = auto()      # Finals ranking, rank-rank
    PP_ELIM = auto()       # Single elimination, percent-percent
    PP_MULTI = auto()      # Multiple elimination, percent-percent
    PP_FINAL = auto()      # Finals ranking, percent-percent


# =============================================================================
# CORE RANKING FUNCTIONS
# These are the authoritative implementations used everywhere.
# =============================================================================

@njit(cache=True)
def ranks_1d(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to ranks where 1 = best (highest score).

    Uses COMPETITION RANKING: ties share the same rank, next rank skips.
    Example: [30, 30, 28] -> [1, 1, 3] (not [1, 2, 3])

    This is the authoritative ranking function used by all premise logic.
    """
    n = len(scores)
    order = np.argsort(-scores)  # descending order
    ranks = np.empty(n, dtype=np.float64)

    current_rank = 1
    for i in range(n):
        if i > 0 and scores[order[i]] < scores[order[i - 1]]:
            current_rank = i + 1  # skip to position-based rank
        ranks[order[i]] = current_rank

    return ranks


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """Python wrapper for ranks_1d (for non-numba contexts)."""
    return ranks_1d(scores.astype(np.float64))


def scores_to_percent(scores: np.ndarray) -> np.ndarray:
    """Convert scores to percentages of total."""
    total = scores.sum()
    if total <= 0:
        return np.ones(len(scores)) / len(scores)
    return scores / total


# =============================================================================
# COMBINATION FUNCTIONS
# =============================================================================

def combine_by_rank(judge_scores: np.ndarray, fan_votes: np.ndarray) -> np.ndarray:
    """
    Rank-based combination (Seasons 1-2, 28-34).

    - Convert both to ranks (1 = best)
    - Sum ranks
    - LOWER sum = BETTER

    Returns: Combined rank sums (lower = better)
    """
    judge_ranks = scores_to_ranks(judge_scores)
    fan_ranks = scores_to_ranks(fan_votes)
    return judge_ranks + fan_ranks


def combine_by_percent(judge_scores: np.ndarray, fan_votes: np.ndarray) -> np.ndarray:
    """
    Percent-based combination (Seasons 3-27).

    - Convert judge scores to percentages
    - Fan votes are already percentages (sum to 1)
    - Sum percentages
    - HIGHER sum = BETTER

    Returns: Combined percentages (higher = better)
    """
    judge_pct = scores_to_percent(judge_scores)
    fan_pct = fan_votes  # already normalized
    return judge_pct + fan_pct


# =============================================================================
# TIEBREAKER CONSTANT
# Used to break ties when combined scores are equal.
# Higher fan votes = better position (survives elimination).
# =============================================================================

TIEBREAKER_WEIGHT = 1e-9


# =============================================================================
# PREMISE TYPE ASSIGNMENT
# =============================================================================

def get_premise_type(season: int, n_eliminated: int, is_final: bool) -> PremiseType:
    """
    Determine the premise type based on season and elimination count.

    Season groupings:
    - S1-2: Rank-Rank (RR)
    - S3-27: Percent-Percent (PP)
    - S28-34: Rank-Rank (RR)

    Bottom-2 rule applies to certain seasons (judge saves one).
    """
    is_multi = n_eliminated > 1

    # Seasons 1-2: Rank-Rank
    if season <= 2:
        if is_final:
            return PremiseType.RR_FINAL
        elif is_multi:
            return PremiseType.RR_MULTI
        else:
            return PremiseType.RR_ELIM

    # Seasons 3-27: Percent-Percent
    elif season <= 27:
        if is_final:
            return PremiseType.PP_FINAL
        elif is_multi:
            return PremiseType.PP_MULTI
        else:
            return PremiseType.PP_ELIM

    # Seasons 28-34: Rank-Rank
    else:
        if is_final:
            return PremiseType.RR_FINAL
        elif is_multi:
            return PremiseType.RR_MULTI
        else:
            return PremiseType.RR_ELIM


# =============================================================================
# VALIDATION FUNCTIONS (Python - for verification and hull validity)
# =============================================================================

def validate_point(
    point: np.ndarray,
    contestants: List[str],
    judge_scores: np.ndarray,
    eliminated: Set[str],
    placements: Optional[np.ndarray],
    season: int,
    is_final: bool,
) -> bool:
    """
    Validate a single vote distribution point against premise constraints.

    This is the authoritative validation function. Uses the same logic
    as the numba-jitted batch functions in this module.

    Args:
        point: Fan vote distribution (sums to 1)
        contestants: List of contestant names
        judge_scores: Judge scores array
        eliminated: Set of eliminated contestant names
        placements: Final placements array (for finals only)
        season: Season number
        is_final: Whether this is a final episode

    Returns:
        True if the point produces the correct outcome
    """
    n_elim = len(eliminated)
    premise_type = get_premise_type(season, n_elim, is_final)

    fan_votes = point.astype(np.float64)
    judge = judge_scores.astype(np.float64)

    # Get indices of eliminated contestants
    elim_indices = np.array([contestants.index(name) for name in eliminated], dtype=np.int64)

    if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI,
                        PremiseType.RR_B2, PremiseType.RR_B2_MULTI,
                        PremiseType.RR_FINAL):
        # Rank-based premises
        judge_ranks = ranks_1d(judge)
        fan_ranks = ranks_1d(fan_votes)
        combined = judge_ranks + fan_ranks

        # Tiebreaker: higher fan votes = better (lower sort key)
        sort_key = combined - fan_votes * TIEBREAKER_WEIGHT

        if is_final and placements is not None:
            # RR_FINAL: check if predicted order matches actual placements
            expected_order = np.argsort(placements).astype(np.int64)
            pred_order = np.argsort(sort_key)
            return np.array_equal(pred_order, expected_order)

        elif premise_type in (PremiseType.RR_B2, PremiseType.RR_B2_MULTI):
            # Bottom-2 rule: eliminated must be in bottom N
            n_bottom = max(2, n_elim)
            order = np.argsort(-sort_key)  # worst first
            bottom = set(order[:n_bottom])
            return all(idx in bottom for idx in elim_indices)

        else:
            # RR_ELIM/RR_MULTI: worst N are eliminated
            order = np.argsort(-sort_key)  # worst first
            worst = set(order[:n_elim])
            return worst == set(elim_indices)

    else:
        # Percent-based premises
        judge_pct = judge / judge.sum()
        fan_pct = fan_votes  # already sums to 1
        combined = judge_pct + fan_pct

        if is_final and placements is not None:
            # PP_FINAL: higher combined = better
            expected_order = np.argsort(placements).astype(np.int64)
            pred_order = np.argsort(-combined)  # best first
            return np.array_equal(pred_order, expected_order)

        else:
            # PP_ELIM/PP_MULTI: lowest combined are eliminated
            order = np.argsort(combined)  # worst first
            worst = set(order[:n_elim])
            return worst == set(elim_indices)


# =============================================================================
# PREDICTION FUNCTIONS (for diagnostics and debugging)
# =============================================================================

@dataclass
class PredictionResult:
    """Result of predicting elimination/ranking outcome."""
    eliminated: Set[str]
    bottom_n: List[str]
    ranking: List[str]
    combined_scores: np.ndarray


def predict_outcome(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    season: int,
    n_elim: int,
    is_final: bool,
) -> PredictionResult:
    """
    Predict the elimination/ranking outcome for given votes.

    Uses the same logic as validate_point.

    Args:
        contestants: List of contestant names
        judge_scores: Judge scores array
        fan_votes: Fan vote distribution (sums to 1)
        season: Season number
        n_elim: Number to eliminate
        is_final: Whether this is a final episode

    Returns:
        PredictionResult with predicted outcome
    """
    premise_type = get_premise_type(season, n_elim, is_final)
    fan = fan_votes.astype(np.float64)
    judge = judge_scores.astype(np.float64)

    if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI,
                        PremiseType.RR_B2, PremiseType.RR_B2_MULTI,
                        PremiseType.RR_FINAL):
        # Rank-based
        judge_ranks = ranks_1d(judge)
        fan_ranks = ranks_1d(fan)
        combined = judge_ranks + fan_ranks
        sort_key = combined - fan * TIEBREAKER_WEIGHT

        if is_final:
            # Ranking: lowest sort_key = 1st place
            order = np.argsort(sort_key)
            ranking = [contestants[i] for i in order]
            return PredictionResult(
                eliminated=set(),
                bottom_n=[],
                ranking=ranking,
                combined_scores=combined,
            )
        else:
            # Elimination: highest sort_key = eliminated
            order = np.argsort(-sort_key)
            n_bottom = max(2, n_elim) if premise_type in (PremiseType.RR_B2, PremiseType.RR_B2_MULTI) else n_elim
            bottom_n = [contestants[i] for i in order[:n_bottom]]
            eliminated = set(bottom_n[:n_elim])
            return PredictionResult(
                eliminated=eliminated,
                bottom_n=bottom_n,
                ranking=[],
                combined_scores=combined,
            )
    else:
        # Percent-based
        judge_pct = judge / judge.sum()
        fan_pct = fan
        combined = judge_pct + fan_pct

        if is_final:
            # Ranking: highest combined = 1st place
            order = np.argsort(-combined)
            ranking = [contestants[i] for i in order]
            return PredictionResult(
                eliminated=set(),
                bottom_n=[],
                ranking=ranking,
                combined_scores=combined,
            )
        else:
            # Elimination: lowest combined = eliminated
            order = np.argsort(combined)
            eliminated = {contestants[i] for i in order[:n_elim]}
            bottom_n = [contestants[i] for i in order[:n_elim]]
            return PredictionResult(
                eliminated=eliminated,
                bottom_n=bottom_n,
                ranking=[],
                combined_scores=combined,
            )


# =============================================================================
# NUMBA-JITTED BATCH VALIDATION FUNCTIONS (for sampler.py performance)
# These MUST use the same logic as validate_point above.
# =============================================================================

@njit(cache=True)
def _argsort_desc(arr):
    """Argsort descending for 1D array."""
    return np.argsort(-arr)


@njit(parallel=True, cache=True)
def check_rr_elim_batch(fan_votes, judge_ranks, elim_indices, n_elim):
    """
    Check RR elimination for batch of fan vote vectors.

    Args:
        fan_votes: (batch_size, n_contestants) fan vote distributions
        judge_ranks: (n_contestants,) pre-computed judge ranks
        elim_indices: indices of actually eliminated contestants
        n_elim: number eliminated

    Returns:
        Boolean array of shape (batch_size,)
    """
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_ranks = ranks_1d(fan_votes[b])
        combined = judge_ranks + fan_ranks

        # Tiebreaker: higher fan votes = better (lower sort key = survives)
        sort_key = combined - fan_votes[b] * 1e-9
        order = np.argsort(-sort_key)  # worst first
        worst = order[:n_elim]

        # Check if worst matches elim_indices (as sets)
        match = True
        for idx in elim_indices:
            found = False
            for w in worst:
                if w == idx:
                    found = True
                    break
            if not found:
                match = False
                break

        if match:
            for w in worst:
                found = False
                for idx in elim_indices:
                    if w == idx:
                        found = True
                        break
                if not found:
                    match = False
                    break

        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def check_pp_elim_batch(fan_votes, judge_pct, elim_indices, n_elim):
    """Check PP elimination for batch."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_pct = fan_votes[b]
        combined = judge_pct + fan_pct

        # Lowest combined = worst
        order = np.argsort(combined)
        worst = order[:n_elim]

        match = True
        for idx in elim_indices:
            found = False
            for w in worst:
                if w == idx:
                    found = True
                    break
            if not found:
                match = False
                break

        if match:
            for w in worst:
                found = False
                for idx in elim_indices:
                    if w == idx:
                        found = True
                        break
                if not found:
                    match = False
                    break

        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def check_rr_b2_batch(fan_votes, judge_ranks, elim_indices, n_bottom):
    """Check RR bottom-2 rule. Eliminated must be IN bottom N."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_ranks = ranks_1d(fan_votes[b])
        combined = judge_ranks + fan_ranks
        sort_key = combined - fan_votes[b] * 1e-9
        order = np.argsort(-sort_key)
        bottom = order[:n_bottom]

        # All elim_indices must be in bottom
        match = True
        for idx in elim_indices:
            found = False
            for w in bottom:
                if w == idx:
                    found = True
                    break
            if not found:
                match = False
                break

        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def check_rr_final_batch(fan_votes, judge_ranks, expected_order):
    """Check RR final ranking."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_ranks = ranks_1d(fan_votes[b])
        combined = judge_ranks + fan_ranks
        sort_key = combined - fan_votes[b] * 1e-9
        pred_order = np.argsort(sort_key)  # best first (lowest sort_key)

        match = True
        for i in range(n):
            if pred_order[i] != expected_order[i]:
                match = False
                break
        valid[b] = match

    return valid


@njit(parallel=True, cache=True)
def check_pp_final_batch(fan_votes, judge_pct, expected_order):
    """Check PP final ranking."""
    batch_size, n = fan_votes.shape
    valid = np.zeros(batch_size, dtype=np.bool_)

    for b in prange(batch_size):
        fan_pct = fan_votes[b]
        combined = judge_pct + fan_pct
        pred_order = np.argsort(-combined)  # best first (highest combined)

        match = True
        for i in range(n):
            if pred_order[i] != expected_order[i]:
                match = False
                break
        valid[b] = match

    return valid


# =============================================================================
# JIT WARMUP
# =============================================================================

_jit_warmed = False

def warmup_jit():
    """Trigger JIT compilation with dummy data."""
    global _jit_warmed
    if _jit_warmed:
        return

    # Small dummy arrays
    dummy_votes = np.random.dirichlet(np.ones(5), size=10)
    dummy_judge = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dummy_elim = np.array([0], dtype=np.int64)
    dummy_order = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    ranks_1d(dummy_judge)
    check_rr_elim_batch(dummy_votes, ranks_1d(dummy_judge), dummy_elim, 1)
    check_pp_elim_batch(dummy_votes, dummy_judge / dummy_judge.sum(), dummy_elim, 1)
    check_rr_b2_batch(dummy_votes, ranks_1d(dummy_judge), dummy_elim, 2)
    check_rr_final_batch(dummy_votes, ranks_1d(dummy_judge), dummy_order)
    check_pp_final_batch(dummy_votes, dummy_judge / dummy_judge.sum(), dummy_order)

    _jit_warmed = True
