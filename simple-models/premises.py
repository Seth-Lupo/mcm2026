"""
DWTS Voting Premises - Exact Show Rules from MCM 2026 Problem C

Historical voting methods:
- Seasons 1-2: RANK combination (judge rank + fan rank)
- Seasons 3-27: PERCENT combination (judge% + fan%)
- Seasons 28-34: RANK combination + BOTTOM TWO judge vote

Premise Types:
- RR_ELIM: Rank-rank, lowest combined eliminated (S1-2)
- PP_ELIM: Percent-percent, lowest combined eliminated (S3-27)
- RR_B2: Rank-rank, bottom 2 then judges vote (S28-34)
- RR_FINAL: Rank-rank finals ranking (S1-2, S28-34)
- PP_FINAL: Percent-percent finals ranking (S3-27)
- MULTI variants for double eliminations
"""
import numpy as np
from typing import Set, List, Optional
from dataclasses import dataclass
from enum import Enum, auto


class PremiseType(Enum):
    """Voting premise types based on DWTS rules."""
    # Seasons 1-2: Rank-Rank
    RR_ELIM = auto()       # Single elimination, lowest rank sum out
    RR_MULTI = auto()      # Multiple elimination
    RR_FINAL = auto()      # Finals ranking

    # Seasons 3-27: Percent-Percent
    PP_ELIM = auto()       # Single elimination, lowest percent sum out
    PP_MULTI = auto()      # Multiple elimination
    PP_FINAL = auto()      # Finals ranking

    # Seasons 28-34: Rank-Rank with Bottom Two Judge Vote
    RR_B2 = auto()         # Bottom 2 identified, judges pick one to eliminate
    RR_B2_MULTI = auto()   # Multiple bottom 2 scenarios


# =============================================================================
# COMBINATION FUNCTIONS
# =============================================================================

def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to ranks where 1 = best (highest score).

    Example: scores [25, 20, 21, 26] -> ranks [2, 4, 3, 1]
    """
    # argsort(-scores) gives indices that would sort descending
    # argsort of that gives each element's rank position
    return (np.argsort(np.argsort(-scores)) + 1).astype(float)


def scores_to_percent(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to percentages of total.

    Example: scores [29, 28, 30, 30] with total 117
    -> percents [24.8%, 23.9%, 25.6%, 25.6%]
    """
    total = scores.sum()
    if total <= 0:
        return np.ones(len(scores)) / len(scores)
    return scores / total


def combine_by_rank(judge_scores: np.ndarray, fan_votes: np.ndarray) -> np.ndarray:
    """
    Rank-based combination (Seasons 1-2, 28-34).

    - Convert both to ranks (1 = best)
    - Sum ranks
    - LOWER sum = BETTER (stays in competition)

    Returns: Combined rank sums (lower = better)
    """
    judge_ranks = scores_to_ranks(judge_scores)
    fan_ranks = scores_to_ranks(fan_votes)
    return judge_ranks + fan_ranks


def combine_by_percent(judge_scores: np.ndarray, fan_votes: np.ndarray) -> np.ndarray:
    """
    Percent-based combination (Seasons 3-27).

    - Convert both to percentages of total
    - Sum percentages
    - HIGHER sum = BETTER (stays in competition)

    Returns: Combined percentages (higher = better)
    """
    judge_pct = scores_to_percent(judge_scores)
    fan_pct = scores_to_percent(fan_votes)
    return judge_pct + fan_pct


# =============================================================================
# OUTCOME PREDICTION
# =============================================================================

@dataclass
class PredictedOutcome:
    """Predicted outcome from applying a premise."""
    eliminated: Set[str]           # Names eliminated
    bottom_n: List[str]            # Bottom N contestants (for B2 rule)
    ranking: List[str]             # Full ranking, best to worst
    combined_scores: np.ndarray    # Raw combined scores


def predict_rr_elimination(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    n_elim: int = 1,
) -> PredictedOutcome:
    """
    Rank-rank elimination (S1-2, S28-34 without B2 rule).
    Highest rank sum (worst) gets eliminated.
    """
    combined = combine_by_rank(judge_scores, fan_votes)

    # Higher combined = worse (higher rank sum), sort descending for worst first
    order = np.argsort(-combined)

    elim_idx = order[:n_elim]
    eliminated = {contestants[i] for i in elim_idx}

    # Full ranking: best (lowest sum) to worst (highest sum)
    ranking_idx = np.argsort(combined)
    ranking = [contestants[i] for i in ranking_idx]

    # Bottom N for reference
    bottom_n = [contestants[i] for i in order[:max(2, n_elim)]]

    return PredictedOutcome(
        eliminated=eliminated,
        bottom_n=bottom_n,
        ranking=ranking,
        combined_scores=combined,
    )


def predict_pp_elimination(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    n_elim: int = 1,
) -> PredictedOutcome:
    """
    Percent-percent elimination (S3-27).
    Lowest percent sum gets eliminated.
    """
    combined = combine_by_percent(judge_scores, fan_votes)

    # Lower combined = worse, sort ascending for worst first
    order = np.argsort(combined)

    elim_idx = order[:n_elim]
    eliminated = {contestants[i] for i in elim_idx}

    # Full ranking: best (highest %) to worst (lowest %)
    ranking_idx = np.argsort(-combined)
    ranking = [contestants[i] for i in ranking_idx]

    bottom_n = [contestants[i] for i in order[:max(2, n_elim)]]

    return PredictedOutcome(
        eliminated=eliminated,
        bottom_n=bottom_n,
        ranking=ranking,
        combined_scores=combined,
    )


def predict_rr_bottom2(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
) -> PredictedOutcome:
    """
    Rank-rank with bottom-two rule (S28-34).

    Bottom 2 by combined rank sum are identified.
    Judges then vote on which of the bottom 2 to eliminate.
    (We can't predict judge vote, so eliminated person must be IN bottom 2)
    """
    combined = combine_by_rank(judge_scores, fan_votes)

    # Higher combined = worse
    order = np.argsort(-combined)

    # Bottom 2 are the two with highest rank sums
    bottom_2 = [contestants[i] for i in order[:2]]

    # Full ranking
    ranking_idx = np.argsort(combined)
    ranking = [contestants[i] for i in ranking_idx]

    return PredictedOutcome(
        eliminated=set(),  # Unknown - depends on judge vote
        bottom_n=bottom_2,
        ranking=ranking,
        combined_scores=combined,
    )


def predict_rr_final(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
) -> PredictedOutcome:
    """
    Rank-rank finals ranking (S1-2, S28-34).
    Lowest rank sum = 1st place.
    """
    combined = combine_by_rank(judge_scores, fan_votes)
    ranking_idx = np.argsort(combined)  # Lowest sum first = best
    ranking = [contestants[i] for i in ranking_idx]

    return PredictedOutcome(
        eliminated=set(),
        bottom_n=[],
        ranking=ranking,
        combined_scores=combined,
    )


def predict_pp_final(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
) -> PredictedOutcome:
    """
    Percent-percent finals ranking (S3-27).
    Highest percent sum = 1st place.
    """
    combined = combine_by_percent(judge_scores, fan_votes)
    ranking_idx = np.argsort(-combined)  # Highest % first = best
    ranking = [contestants[i] for i in ranking_idx]

    return PredictedOutcome(
        eliminated=set(),
        bottom_n=[],
        ranking=ranking,
        combined_scores=combined,
    )


# =============================================================================
# VALIDATION: Is y_hat (fan votes) consistent with actual outcome?
# =============================================================================

def validate_rr_elimination(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    actual_eliminated: Set[str],
) -> bool:
    """S1-2 RR: Check if predicted elimination matches actual."""
    n_elim = len(actual_eliminated)
    outcome = predict_rr_elimination(contestants, judge_scores, fan_votes, n_elim)
    return outcome.eliminated == actual_eliminated


def validate_pp_elimination(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    actual_eliminated: Set[str],
) -> bool:
    """S3-27 PP: Check if predicted elimination matches actual."""
    n_elim = len(actual_eliminated)
    outcome = predict_pp_elimination(contestants, judge_scores, fan_votes, n_elim)
    return outcome.eliminated == actual_eliminated


def validate_rr_bottom2(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    actual_eliminated: Set[str],
) -> bool:
    """
    S28-34 RR+B2: Check if actual eliminated is IN the bottom 2.

    The prediction is valid if the eliminated person(s) are in the bottom N,
    since judges choose who to eliminate from that group.
    """
    outcome = predict_rr_bottom2(contestants, judge_scores, fan_votes)
    # All actually eliminated must be in the predicted bottom N
    n_bottom = max(2, len(actual_eliminated))

    # Recalculate with proper bottom N
    combined = combine_by_rank(judge_scores, fan_votes)
    order = np.argsort(-combined)
    bottom_n_set = {contestants[i] for i in order[:n_bottom]}

    return actual_eliminated.issubset(bottom_n_set)


def validate_rr_final(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    actual_placements: np.ndarray,
) -> bool:
    """S1-2, S28-34 RR Final: Check if predicted ranking matches actual."""
    outcome = predict_rr_final(contestants, judge_scores, fan_votes)
    expected_order = np.argsort(actual_placements)
    expected_ranking = [contestants[i] for i in expected_order]
    return outcome.ranking == expected_ranking


def validate_pp_final(
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    actual_placements: np.ndarray,
) -> bool:
    """S3-27 PP Final: Check if predicted ranking matches actual."""
    outcome = predict_pp_final(contestants, judge_scores, fan_votes)
    expected_order = np.argsort(actual_placements)
    expected_ranking = [contestants[i] for i in expected_order]
    return outcome.ranking == expected_ranking


# =============================================================================
# PREMISE ASSIGNMENT BY SEASON
# =============================================================================

def get_premise_type(season: int, n_eliminated: int, is_final: bool) -> PremiseType:
    """
    Determine the premise type for a given season/week.

    Rules from MCM Problem C:
    - Seasons 1-2: Rank-Rank
    - Seasons 3-27: Percent-Percent
    - Seasons 28-34: Rank-Rank with Bottom Two judge vote
    """
    if season <= 2:
        # Seasons 1-2: Pure rank-rank
        if is_final:
            return PremiseType.RR_FINAL
        elif n_eliminated > 1:
            return PremiseType.RR_MULTI
        else:
            return PremiseType.RR_ELIM

    elif season <= 27:
        # Seasons 3-27: Percent-percent
        if is_final:
            return PremiseType.PP_FINAL
        elif n_eliminated > 1:
            return PremiseType.PP_MULTI
        else:
            return PremiseType.PP_ELIM

    else:
        # Seasons 28-34: Rank-rank with bottom two
        if is_final:
            return PremiseType.RR_FINAL
        elif n_eliminated > 1:
            return PremiseType.RR_B2_MULTI
        else:
            return PremiseType.RR_B2


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def validate(
    premise_type: PremiseType,
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    actual_eliminated: Optional[Set[str]] = None,
    actual_placements: Optional[np.ndarray] = None,
) -> bool:
    """
    Check if fan_votes are consistent with actual outcome under given premise.
    """
    if premise_type == PremiseType.RR_ELIM:
        return validate_rr_elimination(contestants, judge_scores, fan_votes, actual_eliminated)

    elif premise_type == PremiseType.RR_MULTI:
        return validate_rr_elimination(contestants, judge_scores, fan_votes, actual_eliminated)

    elif premise_type == PremiseType.RR_FINAL:
        return validate_rr_final(contestants, judge_scores, fan_votes, actual_placements)

    elif premise_type == PremiseType.PP_ELIM:
        return validate_pp_elimination(contestants, judge_scores, fan_votes, actual_eliminated)

    elif premise_type == PremiseType.PP_MULTI:
        return validate_pp_elimination(contestants, judge_scores, fan_votes, actual_eliminated)

    elif premise_type == PremiseType.PP_FINAL:
        return validate_pp_final(contestants, judge_scores, fan_votes, actual_placements)

    elif premise_type == PremiseType.RR_B2:
        return validate_rr_bottom2(contestants, judge_scores, fan_votes, actual_eliminated)

    elif premise_type == PremiseType.RR_B2_MULTI:
        return validate_rr_bottom2(contestants, judge_scores, fan_votes, actual_eliminated)

    else:
        raise ValueError(f"Unknown premise type: {premise_type}")


def predict(
    premise_type: PremiseType,
    contestants: List[str],
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    n_elim: int = 1,
) -> PredictedOutcome:
    """Apply premise to get predicted outcome."""
    if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI):
        return predict_rr_elimination(contestants, judge_scores, fan_votes, n_elim)

    elif premise_type == PremiseType.RR_FINAL:
        return predict_rr_final(contestants, judge_scores, fan_votes)

    elif premise_type in (PremiseType.PP_ELIM, PremiseType.PP_MULTI):
        return predict_pp_elimination(contestants, judge_scores, fan_votes, n_elim)

    elif premise_type == PremiseType.PP_FINAL:
        return predict_pp_final(contestants, judge_scores, fan_votes)

    elif premise_type in (PremiseType.RR_B2, PremiseType.RR_B2_MULTI):
        return predict_rr_bottom2(contestants, judge_scores, fan_votes)

    else:
        raise ValueError(f"Unknown premise type: {premise_type}")
