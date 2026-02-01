"""
Week Events Organizer

Transforms main.csv into week-based events for easy iteration.
Each event contains:
- Season, week, premise type
- Active contestants with their judge scores
- Outcome (who was eliminated or final placements)
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Set, Optional, Dict, Any
from enum import Enum

from premises import PremiseType, get_premise_type


DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class WeekEvent:
    """A single week's competition event."""
    season: int
    week: int
    premise_type: PremiseType

    # Contestants active this week
    contestants: List[str]
    judge_scores: np.ndarray  # shape (n_contestants,)

    # Outcome
    eliminated: Set[str] = field(default_factory=set)
    placements: Optional[np.ndarray] = None  # For finals: placement[i] for contestants[i]

    # Metadata
    is_final: bool = False
    n_eliminated: int = 0

    @property
    def n(self) -> int:
        return len(self.contestants)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "season": int(self.season),
            "week": int(self.week),
            "premise_type": self.premise_type.name,
            "contestants": self.contestants,
            "judge_scores": [float(x) for x in self.judge_scores],
            "eliminated": list(self.eliminated),
            "placements": [int(x) for x in self.placements] if self.placements is not None else None,
            "is_final": self.is_final,
            "n_eliminated": int(self.n_eliminated),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeekEvent":
        """Create from dict."""
        return cls(
            season=d["season"],
            week=d["week"],
            premise_type=PremiseType[d["premise_type"]],
            contestants=d["contestants"],
            judge_scores=np.array(d["judge_scores"]),
            eliminated=set(d["eliminated"]),
            placements=np.array(d["placements"]) if d["placements"] else None,
            is_final=d["is_final"],
            n_eliminated=d["n_eliminated"],
        )


def load_main_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load main.csv."""
    path = path or DATA_DIR / "main.csv"
    return pd.read_csv(path)


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


def build_week_events(df: pd.DataFrame) -> List[WeekEvent]:
    """
    Build list of WeekEvent from the main dataframe.

    Processes each season and extracts:
    - Normal elimination weeks
    - Double/multi elimination weeks
    - Finals
    """
    events = []

    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season].copy()

        # Get elimination mapping: week -> set of eliminated names
        elim_by_week: Dict[int, Set[str]] = {}
        for _, row in sdf.iterrows():
            week = extract_elimination_week(row["results"])
            if week:
                if week not in elim_by_week:
                    elim_by_week[week] = set()
                elim_by_week[week].add(row["celebrity_name"])

        # Build events for each week
        for week in range(1, 12):
            # Find active contestants (those with positive judge scores this week)
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

            # Skip if no elimination (might be a special week)
            if not eliminated:
                continue

            # Check if eliminated contestants are in active set
            if not eliminated.issubset(set(contestants)):
                continue

            n_eliminated = len(eliminated)
            premise_type = get_premise_type(season, n_eliminated, is_final=False)

            events.append(WeekEvent(
                season=season,
                week=week,
                premise_type=premise_type,
                contestants=contestants,
                judge_scores=judge_scores,
                eliminated=eliminated,
                is_final=False,
                n_eliminated=n_eliminated,
            ))

        # Build finals event
        finalists = sdf[sdf["placement"].isin([1, 2, 3, 4, 5])].copy()
        if not finalists.empty:
            # Find the last week with scores for finalists
            last_week = None
            for w in range(1, 12):
                has_scores = any(get_judge_total(row, w) > 0 for _, row in finalists.iterrows())
                if has_scores:
                    last_week = w

            if last_week:
                contestants = finalists["celebrity_name"].tolist()
                judge_scores = np.array([get_judge_total(row, last_week) for _, row in finalists.iterrows()])
                placements = finalists["placement"].to_numpy()

                premise_type = get_premise_type(season, 0, is_final=True)

                events.append(WeekEvent(
                    season=season,
                    week=last_week,
                    premise_type=premise_type,
                    contestants=contestants,
                    judge_scores=judge_scores,
                    eliminated=set(),
                    placements=placements,
                    is_final=True,
                    n_eliminated=0,
                ))

    return events


def save_events(events: List[WeekEvent], path: Path) -> None:
    """Save events to JSON."""
    data = [e.to_dict() for e in events]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_events(path: Path) -> List[WeekEvent]:
    """Load events from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [WeekEvent.from_dict(d) for d in data]


def summarize_events(events: List[WeekEvent]) -> None:
    """Print summary of events."""
    by_type = {}
    for e in events:
        t = e.premise_type.name
        if t not in by_type:
            by_type[t] = 0
        by_type[t] += 1

    print(f"Total events: {len(events)}")
    print("\nBy premise type:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")

    print("\nBy season:")
    by_season = {}
    for e in events:
        if e.season not in by_season:
            by_season[e.season] = {"elim": 0, "final": 0}
        if e.is_final:
            by_season[e.season]["final"] += 1
        else:
            by_season[e.season]["elim"] += 1

    for s in sorted(by_season.keys()):
        info = by_season[s]
        print(f"  S{s}: {info['elim']} elim weeks, {info['final']} finals")


def main():
    """Build and save week events."""
    print(f"Loading data from {DATA_DIR / 'main.csv'}")
    df = load_main_data()

    print("Building week events...")
    events = build_week_events(df)

    out_path = DATA_DIR / "week_events.json"
    print(f"Saving to {out_path}")
    save_events(events, out_path)

    print("\n" + "=" * 50)
    summarize_events(events)


if __name__ == "__main__":
    main()
