"""
Convex hull approximation via extreme points in random directions.
Fast for any dimension. Includes Monte Carlo validation of hull.

Uses centralized premise logic from premises.py.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set
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
from premises import validate_point

log = logging.getLogger(__name__)


def _sample_from_hull(vertices: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample points uniformly from the convex hull of vertices.

    Uses Dirichlet sampling for random convex combinations.

    Args:
        vertices: (n_vertices, n_dims) array of hull vertices
        n_samples: Number of points to sample
        rng: Random number generator

    Returns:
        (n_samples, n_dims) array of sampled points
    """
    n_verts = len(vertices)
    if n_verts == 0:
        return np.empty((0, vertices.shape[1] if len(vertices.shape) > 1 else 0))

    if n_verts == 1:
        return np.tile(vertices[0], (n_samples, 1))

    # Sample weights from Dirichlet (uniform over simplex of weights)
    weights = rng.dirichlet(np.ones(n_verts), size=n_samples)

    # Compute convex combinations
    return weights @ vertices


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
    hull_validity: float = 0.0  # fraction of hull that satisfies premise (Monte Carlo)
    hull_validity_samples: int = 0  # number of samples used for Monte Carlo
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
            "hull_validity": round(float(self.hull_validity), p),
            "hull_validity_samples": int(self.hull_validity_samples),
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


def compute_region(sample_result: SampleResult, hull_validity_samples: int = 10000) -> RegionInfo:
    """
    Compute region properties from valid samples.

    Uses extreme points method for fast convex hull approximation.
    Includes Monte Carlo estimation of hull validity (what fraction of the
    convex hull actually satisfies the premise constraints).

    Args:
        sample_result: Result from sampling valid votes
        hull_validity_samples: Number of Monte Carlo samples for hull validity

    Returns:
        RegionInfo with computed properties
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
    rng = np.random.default_rng(cfg.sampling.seed)
    if len(valid) > cfg.hull.max_points:
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

    # Monte Carlo hull validity estimation
    # Sample from the convex hull and check what fraction satisfies the premise
    _progress(f"Monte Carlo hull validity ({hull_validity_samples} samples)...")

    hull_samples = _sample_from_hull(base_info.vertices, hull_validity_samples, rng)

    # Prepare event data for verification
    judge_scores = event.judge_scores.astype(np.float64)
    eliminated = event.eliminated
    placements = event.placements

    n_valid_in_hull = 0
    for pt in hull_samples:
        if validate_point(pt, event.contestants, judge_scores, eliminated, placements, event.season, event.is_final):
            n_valid_in_hull += 1

    base_info.hull_validity = n_valid_in_hull / hull_validity_samples if hull_validity_samples > 0 else 0.0
    base_info.hull_validity_samples = hull_validity_samples

    _progress(f"Done: {n_verts} verts, vol={base_info.volume:.2e}, hull_valid={base_info.hull_validity:.1%}", newline=True)

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
