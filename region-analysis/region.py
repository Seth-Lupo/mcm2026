"""
Convex hull approximation via extreme points in random directions.
Fast for any dimension.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import logging

log = logging.getLogger(__name__)

from sampler import SampleResult
from config import get_config


@dataclass
class RegionInfo:
    """Properties of a valid vote region."""
    season: int
    week: int
    n_contestants: int
    premise_type: str

    # Sampling stats
    n_samples: int
    n_valid: int
    acceptance_rate: float

    # Hull properties
    has_hull: bool = False
    n_vertices: int = 0
    volume: float = 0.0
    centroid: Optional[np.ndarray] = None
    vertices: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        cfg = get_config()
        p = cfg.output.precision

        d = {
            "season": int(self.season),
            "week": int(self.week),
            "n_contestants": int(self.n_contestants),
            "premise_type": str(self.premise_type),
            "n_samples": int(self.n_samples),
            "n_valid": int(self.n_valid),
            "acceptance_rate": round(float(self.acceptance_rate), p),
            "has_hull": bool(self.has_hull),
            "n_vertices": int(self.n_vertices),
            "volume": round(float(self.volume), p + 4),
        }

        if cfg.output.save_centroid and self.centroid is not None:
            d["centroid"] = [round(float(x), p) for x in self.centroid]

        if cfg.output.save_vertices and self.vertices is not None:
            d["vertices"] = [[round(float(x), p) for x in row] for row in self.vertices]

        return d


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

    # Estimate volume using widths in random directions
    widths = []
    for d in directions[:min(n_dirs, dim * 3)]:
        projections = points @ d
        widths.append(projections.max() - projections.min())
    avg_width = np.mean(widths) if widths else 0.0

    # Approximate volume (rough estimate)
    approx_volume = (avg_width ** dim) / (2 ** dim) if dim > 0 else 0.0

    return {
        "vertices": vertices,
        "vertex_indices": vertex_indices,
        "n_vertices": len(vertex_indices),
        "volume": approx_volume,
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
        n_contestants=n,
        premise_type=sample_result.premise_type.name,
        n_samples=sample_result.n_samples,
        n_valid=sample_result.n_valid,
        acceptance_rate=sample_result.acceptance_rate,
    )

    # Need points to compute hull
    if sample_result.n_valid == 0:
        log.info(f"  [HULL] No valid points")
        return base_info

    # Centroid
    base_info.centroid = np.mean(valid, axis=0)

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
    base_info.vertices = hull_points[result["vertex_indices"]]

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
