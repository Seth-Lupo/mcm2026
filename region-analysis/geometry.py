"""
Geometric computations for convex hull analysis.

Provides volume, diameter, and simplex calculations used across
the region analysis pipeline. All dimension thresholds and parameters
are configurable via config.yaml.
"""
import numpy as np
from math import factorial
from typing import Tuple
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from config import get_config


def simplex_volume(n: int) -> float:
    """Volume of the (n-1)-simplex in projected coordinates.

    When we project the probability simplex to (n-1) dims by dropping
    the last coordinate, the Euclidean volume is 1/(n-1)!.

    Args:
        n: Number of dimensions (contestants)

    Returns:
        Volume of the standard simplex in projected coordinates
    """
    if n <= 1:
        return 1.0
    return 1.0 / factorial(n - 1)


def compute_diameter(vertices: np.ndarray) -> float:
    """Compute diameter (max pairwise distance) of vertex set.

    Args:
        vertices: Array of shape (n_vertices, dim)

    Returns:
        Maximum Euclidean distance between any two vertices
    """
    if len(vertices) < 2:
        return 0.0
    distances = pdist(vertices)
    return float(distances.max()) if len(distances) > 0 else 0.0


def compute_volume(vertices: np.ndarray, project_simplex: bool = True) -> float:
    """Compute convex hull volume with automatic fallback.

    Uses exact scipy ConvexHull for low dimensions, fast approximation
    for high dimensions. The threshold is set by config.hull.max_dim_exact_volume.

    Args:
        vertices: Array of shape (n_vertices, dim)
        project_simplex: If True, drops last coordinate (simplex projection)

    Returns:
        Volume of the convex hull
    """
    cfg = get_config()

    if len(vertices) < 2:
        return 0.0

    points = vertices[:, :-1] if project_simplex else vertices
    n_points, dim = points.shape

    if dim == 0 or n_points < dim + 1:
        return 0.0

    # Use approximation for high dimensions
    if dim > cfg.hull.max_dim_exact_volume:
        return _approximate_volume(points, cfg.hull.volume_samples)

    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return _approximate_volume(points, cfg.hull.volume_samples)


def _approximate_volume_fast(vertices: np.ndarray) -> float:
    """Fast volume approximation without building ConvexHull.

    Uses average squared distance from centroid to estimate an effective
    "radius", then computes volume as r^d / d!.

    Args:
        vertices: Array of shape (n_vertices, dim)

    Returns:
        Approximate volume
    """
    n_points, dim = vertices.shape
    if dim == 0 or n_points < dim + 1:
        return 0.0

    centroid = vertices.mean(axis=0)
    diffs = vertices - centroid
    avg_sq_dist = np.mean(np.sum(diffs ** 2, axis=1))
    r = np.sqrt(avg_sq_dist)

    return (r ** dim) / factorial(dim)


def _approximate_volume(vertices: np.ndarray, n_samples: int = 100000) -> float:
    """Approximate convex hull volume via Monte Carlo.

    For dims <= fast_volume_dim_threshold (config): Uses ConvexHull + Monte Carlo
    For dims > threshold: Uses fast geometric approximation

    Args:
        vertices: Array of shape (n_vertices, dim)
        n_samples: Number of Monte Carlo samples (for moderate dims)

    Returns:
        Approximate volume
    """
    cfg = get_config()
    n_points, dim = vertices.shape

    if dim == 0 or n_points < dim + 1:
        return 0.0

    # For high dims, skip expensive ConvexHull
    if dim > cfg.hull.fast_volume_dim_threshold:
        return _approximate_volume_fast(vertices)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    widths = maxs - mins

    if np.any(widths == 0):
        return 0.0

    box_volume = np.prod(widths)
    rng = np.random.default_rng(cfg.sampling.seed)
    samples = rng.uniform(mins, maxs, size=(n_samples, dim))

    try:
        hull = ConvexHull(vertices)
        samples_h = np.hstack([samples, np.ones((n_samples, 1))])
        inside = np.all(samples_h @ hull.equations.T <= 1e-10, axis=1)
        fraction_inside = inside.sum() / n_samples
        return box_volume * fraction_inside
    except Exception:
        return _approximate_volume_fast(vertices)


def find_extreme_points(
    points: np.ndarray,
    progress_fn=None,
) -> dict:
    """Find extreme points of a point cloud using random directions.

    Vectorized implementation for performance. Finds points that are
    maximal/minimal when projected onto random unit directions.

    Args:
        points: Array of shape (n_points, dim)
        progress_fn: Optional callback for progress updates

    Returns:
        Dict with keys: vertices, vertex_indices, n_vertices, volume, diameter
    """
    cfg = get_config()
    n_points, dim = points.shape
    rng = np.random.default_rng(cfg.sampling.seed)

    n_dirs = cfg.hull.n_directions
    directions = rng.normal(size=(n_dirs, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    if progress_fn:
        progress_fn(f"{n_points} pts, {dim}D | Projecting {n_dirs} dirs...")

    # Vectorized projection
    all_projections = points @ directions.T

    # Find extremes in all directions
    max_indices = np.argmax(all_projections, axis=0)
    min_indices = np.argmin(all_projections, axis=0)
    vertex_indices = np.unique(np.concatenate([max_indices, min_indices]))

    # Add axis-aligned extremes
    if cfg.hull.include_axis_extremes:
        axis_max = np.argmax(points, axis=0)
        axis_min = np.argmin(points, axis=0)
        vertex_indices = np.unique(np.concatenate([
            vertex_indices, axis_max, axis_min
        ]))

    vertices = points[vertex_indices]
    n_verts = len(vertex_indices)

    if progress_fn:
        progress_fn(f"{n_points} pts, {dim}D | {n_verts} extremes | Volume...")

    volume = compute_volume(vertices, project_simplex=False)

    if progress_fn:
        progress_fn(f"{n_points} pts, {dim}D | {n_verts} extremes | Diameter...")

    diameter = compute_diameter(vertices)

    return {
        "vertices": vertices,
        "vertex_indices": vertex_indices.tolist(),
        "n_vertices": n_verts,
        "volume": volume,
        "diameter": diameter,
    }
