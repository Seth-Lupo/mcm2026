"""
Convex hull approximation via extreme points in random directions.
Fast for any dimension.
"""
import numpy as np
from math import factorial, sqrt
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import logging
from scipy.spatial import ConvexHull

log = logging.getLogger(__name__)

from sampler import SampleResult
from config import get_config


def _progress(msg: str, newline: bool = False) -> None:
    """Print progress on a single line, overwriting previous."""
    end = "\n" if newline else ""
    print(f"\r  [HULL] {msg:<70}", end=end, flush=True)


def simplex_volume(n: int) -> float:
    """Volume of the (n-1)-simplex in projected coordinates.

    When we project the probability simplex to (n-1) dims by dropping the last
    coordinate, the Euclidean volume is 1/(n-1)!. This matches what ConvexHull
    computes on the projected points.
    """
    if n <= 1:
        return 1.0
    return 1.0 / factorial(n - 1)


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
                [round(float(lo), p), round(float(hi), p)]
                for lo, hi in self.dim_bounds
            ]

        if cfg.output.save_vertices and self.vertices is not None:
            region_info["vertices"] = [[round(float(x), p) for x in row] for row in self.vertices]

        return {
            "event": event_info,
            "sampling": sampling_info,
            "region": region_info,
        }


def compute_diameter(vertices: np.ndarray) -> float:
    """Compute diameter (max pairwise distance) of vertices."""
    if len(vertices) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    distances = pdist(vertices)  # All pairwise distances, vectorized
    return float(distances.max()) if len(distances) > 0 else 0.0


def compute_volume(vertices: np.ndarray) -> float:
    """
    Compute convex hull volume.
    Uses exact scipy.spatial.ConvexHull for low dimensions,
    falls back to approximation for high dimensions.
    """
    cfg = get_config()
    n_points, dim = vertices.shape

    # Need at least dim+1 points to form a hull in dim dimensions
    if n_points < dim + 1:
        return 0.0

    # Use approximation for high dimensions (ConvexHull is exponential)
    if dim > cfg.hull.max_dim_exact_volume:
        return _approximate_volume(vertices, cfg.hull.volume_samples)

    try:
        hull = ConvexHull(vertices)
        return hull.volume
    except Exception as e:
        log.debug(f"ConvexHull failed ({e}), using approximation")
        return _approximate_volume(vertices, cfg.hull.volume_samples)


def _approximate_volume_fast(vertices: np.ndarray) -> float:
    """Fast volume approximation WITHOUT building ConvexHull.

    Uses average squared distance from centroid to estimate "radius",
    then computes volume of equivalent d-ball.

    For convex hulls, this correlates well with actual volume.
    """
    n_points, dim = vertices.shape
    if dim == 0 or n_points < dim + 1:
        return 0.0

    # Centroid
    centroid = vertices.mean(axis=0)

    # Average squared distance from centroid
    diffs = vertices - centroid
    avg_sq_dist = np.mean(np.sum(diffs ** 2, axis=1))

    # Effective "radius"
    r = np.sqrt(avg_sq_dist)

    # Volume of d-dimensional ball: π^(d/2) / Γ(d/2 + 1) * r^d
    # Simplified: use r^d scaled by a dimension-dependent factor
    # For a simplex, volume ≈ r^d / d! (rough but fast)
    return (r ** dim) / factorial(dim)


def _approximate_volume(vertices: np.ndarray, n_samples: int = 100000) -> float:
    """Approximate convex hull volume.

    For dims <= 6: Uses ConvexHull + Monte Carlo (accurate but slow for high dims)
    For dims > 6: Uses fast geometric approximation (less accurate but instant)
    """
    n_points, dim = vertices.shape
    if dim == 0 or n_points < dim + 1:
        return 0.0

    # For high dimensions, skip expensive ConvexHull entirely
    if dim > 6:
        return _approximate_volume_fast(vertices)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    widths = maxs - mins
    if np.any(widths == 0):
        return 0.0

    box_volume = np.prod(widths)

    rng = np.random.default_rng(42)
    samples = rng.uniform(mins, maxs, size=(n_samples, dim))

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        samples_h = np.hstack([samples, np.ones((n_samples, 1))])
        inside = np.all(samples_h @ hull.equations.T <= 1e-10, axis=1)
        fraction_inside = inside.sum() / n_samples
        return box_volume * fraction_inside
    except Exception:
        return _approximate_volume_fast(vertices)


def find_extreme_points(points: np.ndarray) -> dict:
    """
    Find extreme points of a point cloud using random directions.
    Fast approximate convex hull for any dimension.

    Fully vectorized for performance.
    """
    cfg = get_config()
    n_points, dim = points.shape
    rng = np.random.default_rng(42)

    # Generate random unit directions
    n_dirs = cfg.hull.n_directions
    directions = rng.normal(size=(n_dirs, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    _progress(f"{n_points} pts, {dim}D | Projecting {n_dirs} dirs...")

    # Vectorized: ONE matrix multiply instead of n_dirs separate ones
    all_projections = points @ directions.T  # (n_points, n_dirs)

    # Vectorized argmax/argmin across all directions at once
    max_indices = np.argmax(all_projections, axis=0)
    min_indices = np.argmin(all_projections, axis=0)

    # Combine unique indices
    vertex_indices = np.unique(np.concatenate([max_indices, min_indices]))

    # Also add axis-aligned extremes (vectorized)
    if cfg.hull.include_axis_extremes:
        axis_max = np.argmax(points, axis=0)
        axis_min = np.argmin(points, axis=0)
        vertex_indices = np.unique(np.concatenate([
            vertex_indices, axis_max, axis_min
        ]))

    vertices = points[vertex_indices]
    n_verts = len(vertex_indices)

    _progress(f"{n_points} pts, {dim}D | {n_verts} extremes | Volume...")

    # Compute exact volume (with fallback)
    volume = compute_volume(vertices)

    _progress(f"{n_points} pts, {dim}D | {n_verts} extremes | Diameter...")

    # Diameter
    diameter = compute_diameter(vertices)

    return {
        "vertices": vertices,
        "vertex_indices": vertex_indices.tolist(),
        "n_vertices": n_verts,
        "volume": volume,
        "diameter": diameter,
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
        indices = np.random.choice(len(valid), cfg.hull.max_points, replace=False)
        hull_points = valid[indices]
    else:
        hull_points = valid

    # Project to (n-1) dims (simplex constraint)
    projected = hull_points[:, :-1]

    result = find_extreme_points(projected)
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
