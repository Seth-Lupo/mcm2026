"""
Configuration loader for region analysis.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class SamplingConfig:
    n_samples: int = 100000
    seed: int = 42


@dataclass
class HullConfig:
    n_directions: int = 200
    max_points: int = 2000
    include_axis_extremes: bool = True
    max_dim_exact_volume: int = 6
    volume_samples: int = 100000
    fast_volume_dim_threshold: int = 6


@dataclass
class OutputConfig:
    save_vertices: bool = True
    save_centroid: bool = True
    precision: int = 15  # max ~15-17 significant digits for float64


@dataclass
class BackprojectionConfig:
    method: str = "lp"
    n_redistribution_samples: int = 1000
    tolerance: float = 1e-12
    bounds_padding: float = 1e-9
    edge_intersection_tol: float = 1e-10


@dataclass
class ParallelConfig:
    n_workers: int = 8
    batch_size: int = 100


@dataclass
class VotingConfig:
    votes_per_viewer: int = 3


@dataclass
class NumericalConfig:
    epsilon: float = 1e-15
    jitter: float = 1e-12
    clamp_probabilities: bool = True


@dataclass
class Config:
    sampling: SamplingConfig
    hull: HullConfig
    output: OutputConfig
    backprojection: BackprojectionConfig
    parallel: ParallelConfig
    voting: VotingConfig
    numerical: NumericalConfig


def _convert_floats(d: dict) -> dict:
    """Convert string representations of floats to actual floats.

    YAML sometimes parses scientific notation like 1e-30 as strings.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v
        else:
            result[k] = v
    return result


def load_config(path: Optional[Path] = None) -> Config:
    """Load config from yaml file."""
    path = path or CONFIG_PATH

    if not path.exists():
        # Return defaults
        return Config(
            sampling=SamplingConfig(),
            hull=HullConfig(),
            output=OutputConfig(),
            backprojection=BackprojectionConfig(),
            parallel=ParallelConfig(),
            voting=VotingConfig(),
            numerical=NumericalConfig(),
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        sampling=SamplingConfig(**_convert_floats(data.get("sampling", {}))),
        hull=HullConfig(**_convert_floats(data.get("hull", {}))),
        output=OutputConfig(**_convert_floats(data.get("output", {}))),
        backprojection=BackprojectionConfig(**_convert_floats(data.get("backprojection", {}))),
        parallel=ParallelConfig(**_convert_floats(data.get("parallel", {}))),
        voting=VotingConfig(**_convert_floats(data.get("voting", {}))),
        numerical=NumericalConfig(**_convert_floats(data.get("numerical", {}))),
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or load config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
