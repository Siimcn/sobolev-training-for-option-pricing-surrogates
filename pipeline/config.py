from dataclasses import dataclass, field

from typing import Tuple

from surrogate_modeling.training_config import TrainingConfig

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "MarketConfig",
    "NetworkConfig",
    "TrainingConfig",
]


@dataclass(frozen=True)
class MarketConfig:

    ticker: str = "AAPL"
    max_maturities: int = 20

    cache_path: str = "data/aapl_chain.json"

    # snapshot the chain so runs are reproducible and work outside US
    # market hours; delete the file to refetch
    use_cache: bool = True

    initial_rate: float = 0.05
    initial_sigma: float = 0.20


@dataclass(frozen=True)
class DataConfig:

    n_samples: int = 1200
    sobolev_order: int = 2
    train_fraction: float = 0.8

    preview_sample_indices: Tuple[int, ...] = (0, 50, 100)
    preview_num_paths: int = 100


@dataclass(frozen=True)
class NetworkConfig:

    architecture: str = "MLP"
    seed: int = 42

    in_size: int = 5
    out_size: int = 1
    width_size: int = 128
    depth: int = 5


def _training_config() -> TrainingConfig:
    """All thirteen fields, so none falls back to a library default."""

    return TrainingConfig(
        learning_rate=1e-3,

        lambda_grad=1.0,
        lambda_hessian=0.1,

        epochs=500,
        batch_size=32,

        early_stopping=True,
        patience=50,
        min_delta=1e-4,

        seed=42,
        print_every=10,

        save_best_model=True,
        checkpoint_path="checkpoints/best_model.eqx",

        sobolev_order=2,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """`training` stays a plain TrainingConfig because config.json archives it."""

    market: MarketConfig = field(default_factory=MarketConfig)
    data: DataConfig = field(default_factory=DataConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=_training_config)

    prints: bool = True

    feature_names: Tuple[str, ...] = ("S", "K", "T", "sigma", "r")
    surface_grid_points: int = 50
