"""
Convex hull approximation via extreme points in random directions.
Fast for any dimension.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import logging

from sampler import SampleResult
from config import get_config
from geometry import (
    simplex_volume,
    compute_volume,
    compute_diameter,
    find_extreme_points,
)

log = logging.getLogger(__name__)


def _progress(msg: str, newline: bool = False) -> None:
    """Print progress on a single line, overwriting previous."""
    end = "\n" if newline else ""
    print(f"\r  [HULL] {msg:<70}", end=end, flush=True)


@dataclass
class RegionInfo:
    """Properties of a valid vote region."""
    # Event info
    season: int
    week: int
    is_final: bool
    premise_type: str
    contestants: List[str]
    n_contestants: int

    # Sampling experiment
    n_samples: int
    n_valid: int
    acceptance_rate: float

    # Region properties
    has_hull: bool = False
    n_vertices: int = 0
    volume: float = 0.0
    relative_volume: float = 0.0
    diameter: float = 0.0
    centroid: Optional[np.ndarray] = None
    dim_bounds: Optional[np.ndarray] = None  # shape (n, 2) for [min, max] per dim
    vertices: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        cfg = get_config()
        p = cfg.output.precision

        # Event info
        event_info = {
            "season": int(self.season),
            "week": int(self.week),
            "is_final": bool(self.is_final),
            "premise_type": str(self.premise_type),
            "contestants": self.contestants,
            "n_contestants": int(self.n_contestants),
        }

        # Sampling experiment
        sampling_info = {
            "n_samples": int(self.n_samples),
            "n_valid": int(self.n_valid),
            "acceptance_rate": round(float(self.acceptance_rate), p),
        }

        # Region properties
        region_info = {
            "has_hull": bool(self.has_hull),
            "n_vertices": int(self.n_vertices),
            "volume": round(float(self.volume), p + 4),
            "relative_volume": round(float(self.relative_volume), p + 4),
            "diameter": round(float(self.diameter), p),
        }

        if cfg.output.save_centroid and self.centroid is not None:
            region_info["centroid"] = [round(float(x), p) for x in self.centroid]

        if self.dim_bounds is not None:
            region_info["dim_bounds"] = [
                {
                    "min": round(float(lo), p),
                    "max": round(float(hi), p),
                    "delta": round(float(hi - lo), p),
                }
                for lo, hi in self.dim_bounds
            ]

        if cfg.output.save_vertices and self.vertices is not None:
            region_info["vertices"] = [[round(float(x), p) for x in row] for row in self.vertices]

        return {
            "event": event_info,
            "sampling": sampling_info,
            "region": region_info,
        }


def compute_region(sample_result: SampleResult) -> RegionInfo:
    """
    Compute region properties from valid samples.
    Uses extreme points method for fast approximation.
    """
    cfg = get_config()
    event = sample_result.event
    n = event.n
    valid = sample_result.valid_samples

    base_info = RegionInfo(
        season=event.season,
        week=event.week,
        is_final=event.is_final,
        premise_type=sample_result.premise_type.name,
        contestants=event.contestants,
        n_contestants=n,
        n_samples=sample_result.n_samples,
        n_valid=sample_result.n_valid,
        acceptance_rate=sample_result.acceptance_rate,
    )

    # Need points to compute hull
    if sample_result.n_valid == 0:
        _progress("No valid points", newline=True)
        return base_info

    # Centroid and dim_bounds from all valid samples
    base_info.centroid = np.mean(valid, axis=0)
    base_info.dim_bounds = np.column_stack([valid.min(axis=0), valid.max(axis=0)])

    # Need at least a few points for meaningful hull
    if sample_result.n_valid < 3:
        _progress(f"Only {sample_result.n_valid} points, skipping hull", newline=True)
        return base_info

    # Subsample if needed
    if len(valid) > cfg.hull.max_points:
        rng = np.random.default_rng(cfg.sampling.seed)
        indices = rng.choice(len(valid), cfg.hull.max_points, replace=False)
        hull_points = valid[indices]
    else:
        hull_points = valid

    # Project to (n-1) dims (simplex constraint)
    projected = hull_points[:, :-1]

    result = find_extreme_points(projected, progress_fn=_progress)
    n_verts = result['n_vertices']

    base_info.has_hull = True
    base_info.n_vertices = result["n_vertices"]
    base_info.volume = result["volume"]
    base_info.diameter = result["diameter"]
    base_info.vertices = hull_points[result["vertex_indices"]]

    # Relative volume compared to full simplex
    full_simplex_vol = simplex_volume(n)
    base_info.relative_volume = result["volume"] / full_simplex_vol if full_simplex_vol > 0 else 0.0

    _progress(f"Done: {n_verts} verts, vol={base_info.volume:.2e}, rel={base_info.relative_volume:.2%}", newline=True)

    return base_info


def save_regions(regions: List[RegionInfo], path: str) -> None:
    """Save region info to JSON."""
    data = [r.to_dict() for r in regions]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_regions(path: str) -> List[Dict[str, Any]]:
    """Load region info from JSON."""
    with open(path) as f:
        return json.load(f)
