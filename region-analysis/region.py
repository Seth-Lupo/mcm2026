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


def simplex_volume(n: int) -> float:
    """Volume of the (n-1)-simplex (probability simplex in n dimensions)."""
    if n <= 1:
        return 1.0
    return sqrt(n) / factorial(n - 1)


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
    max_dist = 0.0
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


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
        return _approximate_volume(vertices, cfg.hull.volume_directions)

    try:
        hull = ConvexHull(vertices)
        return hull.volume
    except Exception as e:
        log.debug(f"ConvexHull failed ({e}), using approximation")
        return _approximate_volume(vertices, cfg.hull.volume_directions)


def _approximate_volume(vertices: np.ndarray, n_dirs: int = 500) -> float:
    """Approximate volume using random direction widths."""
    n_points, dim = vertices.shape
    if dim == 0:
        return 0.0

    rng = np.random.default_rng(42)
    directions = rng.normal(size=(n_dirs, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    widths = []
    for d in directions:
        projections = vertices @ d
        widths.append(projections.max() - projections.min())

    avg_width = np.mean(widths) if widths else 0.0
    return (avg_width ** dim) / (2 ** dim) if dim > 0 else 0.0


def find_extreme_points(points: np.ndarray) -> dict:
    """
    Find extreme points of a point cloud using random directions.
    Fast approximate convex hull for any dimension.
    """
    cfg = get_config()
    n_points, dim = points.shape
    rng = np.random.default_rng(42)

    # Generate random unit directions
    n_dirs = cfg.hull.n_directions
    directions = rng.normal(size=(n_dirs, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Find extreme points in each direction (min and max)
    vertex_indices = set()
    for d in directions:
        projections = points @ d
        vertex_indices.add(int(np.argmax(projections)))
        vertex_indices.add(int(np.argmin(projections)))

    # Also add axis-aligned extremes
    if cfg.hull.include_axis_extremes:
        for i in range(dim):
            vertex_indices.add(int(np.argmax(points[:, i])))
            vertex_indices.add(int(np.argmin(points[:, i])))

    vertex_indices = list(vertex_indices)
    vertices = points[vertex_indices]

    # Compute exact volume (with fallback)
    volume = compute_volume(vertices)

    # Diameter
    diameter = compute_diameter(vertices)

    return {
        "vertices": vertices,
        "vertex_indices": vertex_indices,
        "n_vertices": len(vertex_indices),
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
        log.info(f"  [HULL] No valid points")
        return base_info

    # Centroid and dim_bounds from all valid samples
    base_info.centroid = np.mean(valid, axis=0)
    base_info.dim_bounds = np.column_stack([valid.min(axis=0), valid.max(axis=0)])

    # Need at least a few points for meaningful hull
    if sample_result.n_valid < 3:
        log.info(f"  [HULL] Only {sample_result.n_valid} points, skipping hull")
        return base_info

    # Subsample if needed
    if len(valid) > cfg.hull.max_points:
        log.info(f"  [HULL] Subsampling {cfg.hull.max_points} of {len(valid)} points")
        indices = np.random.choice(len(valid), cfg.hull.max_points, replace=False)
        hull_points = valid[indices]
    else:
        hull_points = valid

    # Project to (n-1) dims (simplex constraint)
    projected = hull_points[:, :-1]
    dim = projected.shape[1]

    log.info(f"  [HULL] Finding extreme points: {len(projected)} pts, {dim} dims...")
    result = find_extreme_points(projected)
    log.info(f"  [HULL] Found {result['n_vertices']} extreme points")

    base_info.has_hull = True
    base_info.n_vertices = result["n_vertices"]
    base_info.volume = result["volume"]
    base_info.diameter = result["diameter"]
    base_info.vertices = hull_points[result["vertex_indices"]]

    # Relative volume compared to full simplex
    full_simplex_vol = simplex_volume(n)
    base_info.relative_volume = result["volume"] / full_simplex_vol if full_simplex_vol > 0 else 0.0

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
