"""Data loading and processing utilities."""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Set

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class WeekState:
    """State for a single week of competition."""
    season: int
    week: int
    contestants: List[str]
    judge_scores: np.ndarray  # shape (n_contestants,)
    eliminated: Set[str]
    is_final: bool = False
    placements: Optional[np.ndarray] = None  # only for finals

    @property
    def n(self) -> int:
        return len(self.contestants)

    def judge_share(self) -> np.ndarray:
        """Judge scores as proportions summing to 1."""
        total = self.judge_scores.sum()
        return self.judge_scores / total if total > 0 else np.ones(self.n) / self.n

    def judge_ranks(self) -> np.ndarray:
        """Ranks from judge scores (1 = best)."""
        return np.argsort(np.argsort(-self.judge_scores)) + 1


def load_main_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the main DWTS dataset."""
    path = path or DATA_DIR / "main.csv"
    return pd.read_csv(path)


def get_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Filter to a single season."""
    sdf = df[df["season"] == season].copy()
    for c in sdf.columns:
        if c.startswith("week") and "_judge" in c:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
    return sdf.reset_index(drop=True)


def week_judge_cols(week: int) -> List[str]:
    return [f"week{week}_judge{i}_score" for i in range(1, 5)]


def get_judge_total(sdf: pd.DataFrame, week: int) -> pd.Series:
    """Sum of judge scores for a week."""
    cols = [c for c in week_judge_cols(week) if c in sdf.columns]
    return sdf[cols].sum(axis=1, skipna=True)


def get_active_mask(sdf: pd.DataFrame, week: int) -> pd.Series:
    """Boolean mask of contestants active in a week."""
    return get_judge_total(sdf, week) > 0


def get_eliminated(sdf: pd.DataFrame, week: int) -> Set[str]:
    """Set of contestant names eliminated in a given week."""
    pat = rf"Eliminated Week {week}\b"
    matches = sdf["results"].astype(str).str.contains(pat, regex=True, na=False)
    return set(sdf.loc[matches, "celebrity_name"].tolist())


def get_week_state(sdf: pd.DataFrame, week: int) -> Optional[WeekState]:
    """Build WeekState for a given week, or None if invalid."""
    mask = get_active_mask(sdf, week)
    if not mask.any():
        return None

    active = sdf.loc[mask].copy()
    contestants = active["celebrity_name"].tolist()
    scores = get_judge_total(active, week).to_numpy()
    eliminated = get_eliminated(sdf, week)

    # Check if eliminated contestants are in active set
    if eliminated and not eliminated.issubset(set(contestants)):
        return None

    return WeekState(
        season=int(sdf["season"].iloc[0]),
        week=week,
        contestants=contestants,
        judge_scores=scores,
        eliminated=eliminated,
    )


def get_final_state(sdf: pd.DataFrame) -> Optional[WeekState]:
    """Build WeekState for the finals."""
    finalists = sdf[sdf["placement"].isin([1, 2, 3, 4, 5])].copy()
    if finalists.empty:
        return None

    # Find last week with scores
    last_week = None
    for w in range(1, 12):
        if (get_judge_total(finalists, w) > 0).any():
            last_week = w
    if last_week is None:
        return None

    contestants = finalists["celebrity_name"].tolist()
    scores = get_judge_total(finalists, last_week).to_numpy()
    placements = finalists["placement"].to_numpy()

    return WeekState(
        season=int(sdf["season"].iloc[0]),
        week=last_week,
        contestants=contestants,
        judge_scores=scores,
        eliminated=set(),
        is_final=True,
        placements=placements,
    )


def iter_weeks(df: pd.DataFrame, seasons: Optional[List[int]] = None):
    """Iterate over all (season, week) states."""
    if seasons is None:
        seasons = sorted(df["season"].unique())

    for s in seasons:
        sdf = get_season(df, s)
        for w in range(1, 12):
            state = get_week_state(sdf, w)
            if state and state.eliminated:
                yield state
        # Also yield finals
        final = get_final_state(sdf)
        if final:
            yield final
