"""
Shared data structures for region analysis.

Contains dataclasses and parsers used across multiple scripts.
"""
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from config import get_config
from geometry import simplex_volume, compute_volume, compute_diameter, safe_divide


@dataclass
class WeekRegion:
    """Parsed region data for a single week.

    Represents a convex region of valid vote distributions for one
    (season, week) event.
    """
    season: int
    week: int
    is_final: bool
    contestants: List[str]
    n_contestants: int
    premise_type: str
    n_valid: int
    vertices: Optional[np.ndarray]  # shape (n_vertices, n_contestants)
    dim_bounds: Optional[np.ndarray]  # shape (n_contestants, 2)
    centroid: Optional[np.ndarray]
    raw_data: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WeekRegion":
        """Parse from JSON dict."""
        event = d["event"]
        region = d["region"]

        vertices = None
        if "vertices" in region and region["vertices"]:
            vertices = np.array(region["vertices"], dtype=np.float64)

        dim_bounds = None
        if "dim_bounds" in region and region["dim_bounds"]:
            # Handle both old format [[min, max], ...] and new format [{"min":, "max":, "delta":}, ...]
            bounds_data = region["dim_bounds"]
            if bounds_data and isinstance(bounds_data[0], dict):
                dim_bounds = np.array(
                    [[float(b["min"]), float(b["max"])] for b in bounds_data],
                    dtype=np.float64
                )
            else:
                dim_bounds = np.array(bounds_data, dtype=np.float64)

        centroid = None
        if "centroid" in region and region["centroid"]:
            centroid = np.array(region["centroid"], dtype=np.float64)

        return cls(
            season=event["season"],
            week=event["week"],
            is_final=event["is_final"],
            contestants=event["contestants"],
            n_contestants=event["n_contestants"],
            premise_type=event["premise_type"],
            n_valid=d["sampling"]["n_valid"],
            vertices=vertices,
            dim_bounds=dim_bounds,
            centroid=centroid,
            raw_data=d,
        )


def load_regions(path: Path) -> List[WeekRegion]:
    """Load and parse regions from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [WeekRegion.from_dict(d) for d in data]


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    """Save results to JSON with proper formatting."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def recompute_region_stats(
    vertices: np.ndarray,
    original_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """Recompute region statistics from filtered vertices.

    Used after projection filtering to update volume, bounds, etc.

    Args:
        vertices: Filtered vertex array
        original_data: Original region JSON data to update

    Returns:
        Tuple of (updated data dict, filtered volume)
    """
    result = json.loads(json.dumps(original_data))  # deep copy
    cfg = get_config()
    p = cfg.output.precision

    if len(vertices) == 0:
        result["region"]["has_hull"] = False
        result["region"]["n_vertices"] = 0
        result["region"]["vertices"] = []
        result["region"]["dim_bounds"] = None
        result["region"]["centroid"] = None
        result["region"]["volume"] = 0.0
        result["region"]["relative_volume"] = 0.0
        result["region"]["diameter"] = 0.0
        return result, 0.0

    filtered_volume = compute_volume(vertices)
    diameter = compute_diameter(vertices)

    n_contestants = vertices.shape[1]
    full_simplex_vol = simplex_volume(n_contestants)
    relative_volume = safe_divide(filtered_volume, full_simplex_vol, default=0.0)

    result["region"]["n_vertices"] = len(vertices)
    result["region"]["vertices"] = [
        [round(float(x), p) for x in row] for row in vertices
    ]
    result["region"]["centroid"] = [
        round(float(x), p) for x in vertices.mean(axis=0)
    ]
    result["region"]["dim_bounds"] = [
        {
            "min": round(float(vertices[:, i].min()), p),
            "max": round(float(vertices[:, i].max()), p),
            "delta": round(float(vertices[:, i].max() - vertices[:, i].min()), p),
        }
        for i in range(vertices.shape[1])
    ]
    result["region"]["volume"] = round(float(filtered_volume), p + 4)
    result["region"]["relative_volume"] = round(float(relative_volume), p + 4)
    result["region"]["diameter"] = round(float(diameter), p)

    return result, filtered_volume


def find_eliminated(week_n: WeekRegion, week_n1: WeekRegion) -> List[str]:
    """Find contestants eliminated between week N and week N+1."""
    return [c for c in week_n.contestants if c not in week_n1.contestants]


def get_contestant_mapping(
    week_n: WeekRegion,
    week_n1: WeekRegion,
) -> Tuple[List[int], List[int]]:
    """Get index mappings for survivors between weeks.

    Returns:
        Tuple of (survivor_indices_n, survivor_indices_n1)
    """
    survivors_n = []
    survivors_n1 = []

    for i, name in enumerate(week_n.contestants):
        if name in week_n1.contestants:
            survivors_n.append(i)
            survivors_n1.append(week_n1.contestants.index(name))

    return survivors_n, survivors_n1


def get_contestant_indices(
    week_n: WeekRegion,
    week_n1: WeekRegion,
) -> Tuple[List[int], List[int], List[int]]:
    """Get full index mappings between weeks.

    Returns:
        Tuple of (eliminated_indices_n, survivor_indices_n, survivor_indices_n1)
    """
    eliminated = find_eliminated(week_n, week_n1)
    eliminated_indices_n = [week_n.contestants.index(e) for e in eliminated]
    survivor_indices_n, survivor_indices_n1 = get_contestant_mapping(week_n, week_n1)
    return eliminated_indices_n, survivor_indices_n, survivor_indices_n1
