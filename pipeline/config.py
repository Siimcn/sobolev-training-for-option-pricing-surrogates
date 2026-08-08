from dataclasses import dataclass, field

from typing import Optional, Tuple

from surrogate_modeling.data_generation import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
    PRICING_MODELS,
)
from surrogate_modeling.training_config import TrainingConfig

__all__ = [
    "BASKET_BLACK_SCHOLES",
    "BLACK_SCHOLES",
    "BasketConfig",
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

    # Calibration engine. True uses the closed-form Black-Scholes price:
    # fast (~2s) and bitwise reproducible. False prices the same
    # instruments with the Monte Carlo simulator instead, which is ~50x
    # slower, depends on the seed, and inherits the Euler/smoothing bias
    # of the simulation (it shifts sigma by roughly +0.4%). Black-Scholes
    # has a closed form, so True is the better choice here; the Monte
    # Carlo branch exists for models that have none.
    black_scholes_analytic: bool = False

    # only read when black_scholes_analytic is False
    mc_calibration_seed: int = 0
    mc_calibration_paths: int = 50_000
    mc_calibration_steps: int = 50


@dataclass(frozen=True)
class BasketConfig:
    """
    Read only when pricing_model is BASKET_BLACK_SCHOLES.

    The basket structure is fixed here rather than carried in the feature
    vector, so the surrogate's input is [S_1, ..., S_n, K, T]. All assets
    share the calibrated volatility and rate, which means a basket
    surrogate has no vega or rho - only per-asset deltas and gammas.
    """

    n_assets: int = 3
    correlation: float = 0.5

    # None means equal weights
    weights: Optional[Tuple[float, ...]] = None

    num_paths: int = 50_000
    num_steps: int = 50
    label_seed: int = 0


@dataclass(frozen=True)
class DataConfig:

    # what the surrogate learns to price; see PRICING_MODELS
    pricing_model: str = BLACK_SCHOLES

    n_samples: int = 1200
    sobolev_order: int = 2
    train_fraction: float = 0.8

    preview_sample_indices: Tuple[int, ...] = (0, 50, 100)
    preview_num_paths: int = 100


@dataclass(frozen=True)
class NetworkConfig:

    architecture: str = "MLP"
    seed: int = 42

    # None derives the input width from the dataset
    in_size: Optional[int] = None
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
    basket: BasketConfig = field(default_factory=BasketConfig)
    training: TrainingConfig = field(default_factory=_training_config)

    prints: bool = True

    surface_grid_points: int = 50

    def __post_init__(self):
        if self.data.pricing_model not in PRICING_MODELS:
            raise ValueError(
                f"Unknown pricing_model '{self.data.pricing_model}'. "
                f"Expected one of: {', '.join(PRICING_MODELS)}."
            )

        weights = self.basket.weights

        if weights is not None and len(weights) != self.basket.n_assets:
            raise ValueError(
                f"basket.weights has {len(weights)} entries but "
                f"n_assets is {self.basket.n_assets}."
            )

    @property
    def is_basket(self) -> bool:
        return self.data.pricing_model == BASKET_BLACK_SCHOLES

    @property
    def feature_names(self) -> Tuple[str, ...]:
        if self.is_basket:
            spots = tuple(f"S{i + 1}" for i in range(self.basket.n_assets))
            return spots + ("K", "T")

        return ("S", "K", "T", "sigma", "r")
