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
class RefinementConfig:
    enabled: bool = True
    min_acceptance: float = 0.01
    walks_per_seed: int = 50
    steps_per_walk: int = 20
    step_size: float = 0.05
    max_seeds: int = 100


@dataclass
class HullConfig:
    n_directions: int = 200
    max_points: int = 2000
    include_axis_extremes: bool = True
    max_dim_exact_volume: int = 6
    volume_directions: int = 500


@dataclass
class OutputConfig:
    save_vertices: bool = True
    save_centroid: bool = True
    precision: int = 6


@dataclass
class BackprojectionConfig:
    method: str = "lp"
    n_redistribution_samples: int = 1000
    tolerance: float = 1e-9
    bounds_padding: float = 0.001


@dataclass
class Config:
    sampling: SamplingConfig
    refinement: RefinementConfig
    hull: HullConfig
    output: OutputConfig
    backprojection: BackprojectionConfig


def load_config(path: Optional[Path] = None) -> Config:
    """Load config from yaml file."""
    path = path or CONFIG_PATH

    if not path.exists():
        # Return defaults
        return Config(
            sampling=SamplingConfig(),
            refinement=RefinementConfig(),
            hull=HullConfig(),
            output=OutputConfig(),
            backprojection=BackprojectionConfig(),
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        sampling=SamplingConfig(**data.get("sampling", {})),
        refinement=RefinementConfig(**data.get("refinement", {})),
        hull=HullConfig(**data.get("hull", {})),
        output=OutputConfig(**data.get("output", {})),
        backprojection=BackprojectionConfig(**data.get("backprojection", {})),
    )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or load config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
