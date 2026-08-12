from dataclasses import asdict, dataclass, field

from typing import Optional, Tuple

from market_simulation.payoff import available_payoffs

# importing the package registers every problem; see its __init__
import surrogate_modeling.problems  # noqa: F401

from surrogate_modeling.problems import (
    BACHELIER,
    BASKET_BACHELIER,
    BASKET_BLACK_SCHOLES,
    BASKET_HESTON,
    BLACK_SCHOLES,
    HESTON,
)
from surrogate_modeling.pricing_problem import available_problems
from surrogate_modeling.training_config import TOTAL, TrainingConfig

__all__ = [
    "BACHELIER",
    "BASKET_BACHELIER",
    "BASKET_BLACK_SCHOLES",
    "BASKET_HESTON",
    "BLACK_SCHOLES",
    "HESTON",
    "BasketConfig",
    "HestonConfig",
    "DataConfig",
    "ExperimentConfig",
    "MarketConfig",
    "NetworkConfig",
    "PayoffConfig",
    "RiskConfig",
    "SimulationConfig",
    "TrainingConfig",
    "ValidationConfig",
]


@dataclass(frozen=True)
class MarketConfig:

    ticker: str = "AAPL"
    max_maturities: int = 20

    cache_path: str = "data/aapl_chain.json"

    use_cache: bool = True

    initial_rate: float = 0.05
    initial_sigma: float = 0.20

    black_scholes_analytic: bool = True

    mc_calibration_seed: int = 0
    mc_calibration_paths: int = 50_000
    mc_calibration_steps: int = 50


@dataclass(frozen=True)
class PayoffConfig:
    """What is being priced; see market_simulation.payoff.available_payoffs()."""

    name: str = "european_call"

    smooth_fraction: float = 0.05


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo settings for the training labels, shared by every model."""

    num_paths: int = 200_000

    num_steps: int = 50

    label_seed: int = 0
    antithetic: bool = True

    # leave False: one shared stream biases every label the same way
    shared_label_keys: bool = False

    reference_paths: int = 1_000_000


@dataclass(frozen=True)
class BasketConfig:
    """Read by every basket problem."""

    n_assets: int = 3

    # assumed, not calibrated: the chain holds no basket instrument
    correlation: float = 0.5

    # None means equal weights
    weights: Optional[Tuple[float, ...]] = None

    # sorts the spots before pricing; needs an exchangeable basket, so turn
    # it off for custom weights
    symmetrize: bool = True


@dataclass(frozen=True)
class HestonConfig:
    """Read only by the two Heston problems."""

    initial_kappa: float = 2.0
    initial_xi: float = 0.5
    initial_rho: float = -0.7

    parameter_band: float = 0.3

    # separate from SimulationConfig: Heston steps through time, so cost
    # scales with paths * steps and the exact-sampling budget runs out of memory
    num_paths: int = 50_000
    num_steps: int = 64
    reference_paths: int = 200_000


@dataclass(frozen=True)
class DataConfig:

    pricing_model: str = BASKET_HESTON

    n_samples: int = 600
    sobolev_order: int = 2
    train_fraction: float = 0.8

    seed: int = 0

    # floor, not cosmetic: the shortest expiries produce curvature labels
    # thousands of times larger than the rest of the domain
    min_maturity: float = 0.05

    r_spread: float = 0.02
    domain_n_sigma: float = 3.0
    domain_horizon: float = 1.0

    preview_sample_indices: Tuple[int, ...] = (0, 50, 100)
    preview_num_paths: int = 100


@dataclass(frozen=True)
class NetworkConfig:

    architecture: str = "MLP"
    seed: int = 42

    in_size: Optional[int] = None
    out_size: int = 1
    width_size: int = 128
    depth: int = 5


@dataclass(frozen=True)
class ValidationConfig:
    """The independent benchmark run against the trained surrogate."""

    enabled: bool = True

    n_points: int = 128

    seed: int = 12345

    arbitrage_tolerance_sigma: float = 3.0


@dataclass(frozen=True)
class RiskConfig:

    enabled: bool = True

    horizon: float = 1.0
    num_paths: int = 100
    num_steps: int = 252
    seed: int = 0


def _training_config() -> TrainingConfig:
    """Every field set explicitly, so none falls back to a library default."""

    return TrainingConfig(
        learning_rate=1e-3,
        lambda_grad=1.0,
        lambda_hessian=0.1,
        epochs=1000,
        batch_size=32,
        lr_schedule="cosine",
        lr_final_fraction=0.02,
        warmup_epochs=10,
        gradient_clip=1.0,
        # on price+gradient alone, training stops while curvature is still
        # improving and leaves its error 2.6x higher
        selection_metric=TOTAL,
        early_stopping=True,
        patience=200,
        min_delta=1e-6,
        min_delta_relative=1e-3,
        seed=42,
        print_every=25,
        sobolev_order=2,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Nothing here answers "which model is this?" - the stages take a
    PricingProblem built from `data.pricing_model` and ask it instead.
    """

    market: MarketConfig = field(default_factory=MarketConfig)
    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    basket: BasketConfig = field(default_factory=BasketConfig)
    heston: HestonConfig = field(default_factory=HestonConfig)
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

        if self.payoff.name not in available_payoffs():
            raise ValueError(
                f"Unknown payoff '{self.payoff.name}'. "
                f"Expected one of: {', '.join(available_payoffs())}."
            )

        weights = self.basket.weights

        if weights is not None and len(weights) != self.basket.n_assets:
            raise ValueError(
                f"basket.weights has {len(weights)} entries but "
                f"n_assets is {self.basket.n_assets}."
            )

    def to_dict(self, problem=None) -> dict:
        """Everything a run needs to be reproducible, for config.json."""

        data = asdict(self)

        if problem is not None:
            data["derived"] = problem.describe()

        return data
