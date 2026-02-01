"""
Event extraction from main.csv.
Simplified from simple-models/events.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Optional, Dict, Any
import json
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            "season": int(self.season),
            "week": int(self.week),
            "is_final": self.is_final,
            "n_contestants": self.n,
            "contestants": self.contestants,
            "judge_scores": [round(float(x), 2) for x in self.judge_scores],
            "eliminated": list(self.eliminated),
        }
        if self.placements is not None:
            d["placements"] = [int(x) for x in self.placements]
        return d


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


def get_active_contestants(sdf: pd.DataFrame, week: int) -> Dict[str, pd.Series]:
    """Get contestants with non-zero scores for a given week."""
    active = {}
    for _, row in sdf.iterrows():
        score = get_judge_total(row, week)
        if score > 0:
            active[row["celebrity_name"]] = row
    return active


def load_events(path: Optional[Path] = None) -> List[Event]:
    """
    Load all events from main.csv.
    Returns list of Event objects for each (season, week) with eliminations.

    Eliminations are detected by comparing who has scores week-to-week:
    if someone danced in week N but not week N+1, they were eliminated in week N.
    """
    path = path or DATA_DIR / "main.csv"
    df = pd.read_csv(path)
    events = []

    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season].copy()

        # Build week -> active contestants mapping
        active_by_week: Dict[int, Dict[str, pd.Series]] = {}
        for week in range(1, 12):
            active = get_active_contestants(sdf, week)
            if active:
                active_by_week[week] = active

        if not active_by_week:
            continue

        weeks_with_data = sorted(active_by_week.keys())
        last_week = max(weeks_with_data)

        # Detect eliminations by comparing consecutive weeks
        for i, week in enumerate(weeks_with_data):
            active = active_by_week[week]
            contestants = list(active.keys())
            judge_scores = np.array([get_judge_total(active[name], week) for name in contestants])

            # Find who was eliminated this week
            if week == last_week:
                # Last week is the final - no elimination event, handle separately
                continue

            # Find next week with data
            next_week = None
            for w in weeks_with_data[i + 1:]:
                next_week = w
                break

            if next_week is None:
                continue

            next_active = active_by_week[next_week]
            eliminated = set(contestants) - set(next_active.keys())

            if not eliminated:
                # No elimination this week (rare, but possible)
                continue

            events.append(Event(
                season=season,
                week=week,
                contestants=contestants,
                judge_scores=judge_scores,
                eliminated=eliminated,
                is_final=False,
            ))

        # Finals - last week with data
        if last_week in active_by_week:
            active = active_by_week[last_week]
            contestants = list(active.keys())
            judge_scores = np.array([get_judge_total(active[name], last_week) for name in contestants])

            # Get placements for finalists
            finalists_df = sdf[sdf["celebrity_name"].isin(contestants)]
            if not finalists_df.empty and "placement" in finalists_df.columns:
                # Build placement array in same order as contestants
                placements = []
                for name in contestants:
                    row = finalists_df[finalists_df["celebrity_name"] == name]
                    if not row.empty:
                        p = row["placement"].values[0]
                        placements.append(int(p) if pd.notna(p) else 99)
                    else:
                        placements.append(99)
                placements = np.array(placements)

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


def save_events(events: List[Event], path: Optional[Path] = None) -> None:
    """Save events to JSON."""
    path = path or DATA_DIR / "events.json"
    data = [e.to_dict() for e in events]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
