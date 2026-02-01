"""
Event extraction from main.csv.
Simplified from simple-models/events.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Optional, Dict
import logging

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class Event:
    """A single week's competition event."""
    season: int
    week: int
    contestants: List[str]
    judge_scores: np.ndarray
    eliminated: Set[str]
    placements: Optional[np.ndarray] = None
    is_final: bool = False

    @property
    def n(self) -> int:
        return len(self.contestants)

    @property
    def n_eliminated(self) -> int:
        return len(self.eliminated)


def get_judge_total(row: pd.Series, week: int) -> float:
    """Sum judge scores for a contestant in a given week."""
    total = 0.0
    for j in range(1, 5):
        col = f"week{week}_judge{j}_score"
        if col in row.index:
            val = pd.to_numeric(row[col], errors="coerce")
            if not pd.isna(val):
                total += val
    return total


def extract_elimination_week(result: str) -> Optional[int]:
    """Extract week number from result string like 'Eliminated Week 3'."""
    if pd.isna(result):
        return None
    if "Eliminated Week" in str(result):
        try:
            return int(str(result).split("Week")[1].strip())
        except:
            return None
    return None


def load_events(path: Optional[Path] = None) -> List[Event]:
    """
    Load all events from main.csv.
    Returns list of Event objects for each (season, week) with eliminations.
    """
    path = path or DATA_DIR / "main.csv"
    df = pd.read_csv(path)
    events = []

    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season].copy()

        # Build elimination mapping: week -> eliminated names
        elim_by_week: Dict[int, Set[str]] = {}
        for _, row in sdf.iterrows():
            week = extract_elimination_week(row["results"])
            if week:
                if week not in elim_by_week:
                    elim_by_week[week] = set()
                elim_by_week[week].add(row["celebrity_name"])

        # Build events for each week
        for week in range(1, 12):
            active_rows = []
            for _, row in sdf.iterrows():
                score = get_judge_total(row, week)
                if score > 0:
                    active_rows.append(row)

            if not active_rows:
                continue

            contestants = [r["celebrity_name"] for r in active_rows]
            judge_scores = np.array([get_judge_total(r, week) for r in active_rows])
            eliminated = elim_by_week.get(week, set())

            if not eliminated:
                continue
            if not eliminated.issubset(set(contestants)):
                continue

            events.append(Event(
                season=season,
                week=week,
                contestants=contestants,
                judge_scores=judge_scores,
                eliminated=eliminated,
                is_final=False,
            ))

        # Finals
        finalists = sdf[sdf["placement"].isin([1, 2, 3, 4, 5])].copy()
        if not finalists.empty:
            last_week = None
            for w in range(1, 12):
                has_scores = any(get_judge_total(row, w) > 0 for _, row in finalists.iterrows())
                if has_scores:
                    last_week = w

            if last_week:
                contestants = finalists["celebrity_name"].tolist()
                judge_scores = np.array([get_judge_total(row, last_week) for _, row in finalists.iterrows()])
                placements = finalists["placement"].to_numpy()

                events.append(Event(
                    season=season,
                    week=last_week,
                    contestants=contestants,
                    judge_scores=judge_scores,
                    eliminated=set(),
                    placements=placements,
                    is_final=True,
                ))

    return events
