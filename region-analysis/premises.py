"""
Premise validation for region analysis.
Imports core logic from simple-models/premises.py.
"""
import sys
from pathlib import Path
import importlib.util
import logging

log = logging.getLogger(__name__)

# Load simple-models/premises.py directly to avoid name conflict
log.debug("Loading premises from simple-models...")
_premises_path = Path(__file__).parent.parent / "simple-models" / "premises.py"
_spec = importlib.util.spec_from_file_location("simple_premises", _premises_path)
_simple_premises = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_simple_premises)
log.debug("Premises loaded.")

PremiseType = _simple_premises.PremiseType
get_premise_type = _simple_premises.get_premise_type
combine_by_rank = _simple_premises.combine_by_rank
combine_by_percent = _simple_premises.combine_by_percent
scores_to_ranks = _simple_premises.scores_to_ranks
scores_to_percent = _simple_premises.scores_to_percent
import numpy as np
from typing import Set, List


def check_elimination(
    premise_type: PremiseType,
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    contestants: List[str],
    actual_eliminated: Set[str],
) -> bool:
    """
    Check if fan_votes produce the correct elimination under given premise.
    Returns True if the vote vector is valid (produces correct outcome).
    """
    n = len(contestants)
    n_elim = len(actual_eliminated)

    if premise_type in (PremiseType.RR_ELIM, PremiseType.RR_MULTI):
        # Rank-rank: highest sum eliminated
        combined = combine_by_rank(judge_scores, fan_votes)
        order = np.argsort(-combined)  # worst first
        pred_elim = {contestants[i] for i in order[:n_elim]}
        return pred_elim == actual_eliminated

    elif premise_type in (PremiseType.PP_ELIM, PremiseType.PP_MULTI):
        # Percent-percent: lowest sum eliminated
        combined = combine_by_percent(judge_scores, fan_votes)
        order = np.argsort(combined)  # worst first
        pred_elim = {contestants[i] for i in order[:n_elim]}
        return pred_elim == actual_eliminated

    elif premise_type in (PremiseType.RR_B2, PremiseType.RR_B2_MULTI):
        # Bottom-2: eliminated must be in bottom N
        combined = combine_by_rank(judge_scores, fan_votes)
        order = np.argsort(-combined)
        n_bottom = max(2, n_elim)
        bottom_n = {contestants[i] for i in order[:n_bottom]}
        return actual_eliminated.issubset(bottom_n)

    return False


def check_final_ranking(
    premise_type: PremiseType,
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    contestants: List[str],
    actual_placements: np.ndarray,
) -> bool:
    """
    Check if fan_votes produce correct final ranking.
    """
    if premise_type == PremiseType.RR_FINAL:
        combined = combine_by_rank(judge_scores, fan_votes)
        pred_order = np.argsort(combined)  # lowest sum = 1st
    elif premise_type == PremiseType.PP_FINAL:
        combined = combine_by_percent(judge_scores, fan_votes)
        pred_order = np.argsort(-combined)  # highest pct = 1st
    else:
        return False

    pred_ranking = [contestants[i] for i in pred_order]
    expected_order = np.argsort(actual_placements)
    expected_ranking = [contestants[i] for i in expected_order]
    return pred_ranking == expected_ranking
