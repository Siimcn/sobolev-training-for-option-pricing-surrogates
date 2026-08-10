from dataclasses import asdict, dataclass, field

from typing import Optional, Tuple

# imported for the registration side effect: the built-in problems must be
# in the registry before `pricing_model` can be validated against it
import surrogate_modeling.problems  # noqa: F401

from surrogate_modeling.problems import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
)
from surrogate_modeling.pricing_problem import available_problems
from surrogate_modeling.training_config import PRICE_GRADIENT, TrainingConfig

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

    # Sort the spots before pricing. The true price is invariant under
    # permuting them, the MC estimator is not; sorting restores that
    # exactly. Requires an exchangeable basket, so turn it off for
    # custom weights.
    symmetrize: bool = True


@dataclass(frozen=True)
class DataConfig:

    # what the surrogate learns to price; see available_problems()
    pricing_model: str = BASKET_BLACK_SCHOLES

    n_samples: int = 600
    sobolev_order: int = 2
    train_fraction: float = 0.8

    seed: int = 0

    # Floor on the sampled maturity. The shortest market expiries produce
    # curvature labels thousands of times larger than the rest of the
    # domain, which the pooled HVP error then reports almost exclusively.
    min_maturity: float = 0.05

    # Shape of the sampling domain. Spots span a `domain_n_sigma` lognormal
    # move over `domain_horizon` years, matching the exposure simulation;
    # `r_spread` is the band around the calibrated rate, and is only read by
    # models that carry the rate as a feature.
    r_spread: float = 0.02
    domain_n_sigma: float = 3.0
    domain_horizon: float = 1.0

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


@dataclass(frozen=True)
class ValidationConfig:
    """The independent benchmark run against the trained surrogate."""

    enabled: bool = True
    n_points: int = 9

    # unrelated to the label seed, so the benchmark is genuinely independent
    seed: int = 12345


@dataclass(frozen=True)
class RiskConfig:

    enabled: bool = True

    horizon: float = 1.0
    num_paths: int = 100
    num_steps: int = 252
    seed: int = 0


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

        # relative, so the criterion keeps its meaning as the loss shrinks;
        # on price+gradient, because the HVP term carries ~85% of the
        # objective's magnitude and cannot be learned to that precision
        min_delta=1e-6,
        min_delta_relative=1e-3,
        selection_metric=PRICE_GRADIENT,

        seed=42,
        print_every=10,

        save_best_model=True,
        checkpoint_path="checkpoints/best_model.eqx",

        sobolev_order=2,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """
    `training` stays a plain TrainingConfig because config.json archives it.

    Nothing here answers "which model is this?" - the stages take a
    PricingProblem built from `data.pricing_model` and ask it instead.
    """

    market: MarketConfig = field(default_factory=MarketConfig)
    data: DataConfig = field(default_factory=DataConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    basket: BasketConfig = field(default_factory=BasketConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    training: TrainingConfig = field(default_factory=_training_config)

    prints: bool = True

    surface_grid_points: int = 50

    def __post_init__(self):
        if self.data.pricing_model not in available_problems():
            raise ValueError(
                f"Unknown pricing_model '{self.data.pricing_model}'. "
                f"Expected one of: {', '.join(available_problems())}."
            )

        weights = self.basket.weights

        if weights is not None and len(weights) != self.basket.n_assets:
            raise ValueError(
                f"basket.weights has {len(weights)} entries but "
                f"n_assets is {self.basket.n_assets}."
            )

    def to_dict(self, problem=None) -> dict:
        """
        Everything a run needs to be reproducible, for config.json.

        `problem` contributes the feature layout and the sampled domain,
        which are derived from the config rather than stated in it.
        """

        data = asdict(self)

        if problem is not None:
            data["derived"] = problem.describe()

        return data
