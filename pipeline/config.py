from dataclasses import asdict, dataclass, field

from typing import Optional, Tuple

from marktsimulation.payoff import available_payoffs

# imported for the registration side effect: the built-in problems must be
# in the registry before `pricing_model` can be validated against it
import surrogate_modeling.problems  # noqa: F401

from surrogate_modeling.problems import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
)
from surrogate_modeling.pricing_problem import available_problems
from surrogate_modeling.training_config import TOTAL, TrainingConfig

__all__ = [
    "BASKET_BLACK_SCHOLES",
    "BLACK_SCHOLES",
    "BasketConfig",
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

    # snapshot the chain so runs are reproducible and work outside US
    # market hours; delete the file to refetch
    use_cache: bool = True

    initial_rate: float = 0.05
    initial_sigma: float = 0.20

    # Calibration engine. True uses the closed-form Black-Scholes price:
    # fast and bitwise reproducible. False prices the same instruments
    # with the Monte Carlo simulator, which is far slower, depends on the
    # seed, and inherits the simulation's payoff-smoothing bias. Black-
    # Scholes has a closed form, so True is the better choice here; the
    # Monte Carlo branch exists for models that have none.
    black_scholes_analytic: bool = True

    # only read when black_scholes_analytic is False
    mc_calibration_seed: int = 0
    mc_calibration_paths: int = 50_000
    mc_calibration_steps: int = 50


@dataclass(frozen=True)
class PayoffConfig:
    """What is being priced; see marktsimulation.payoff.available_payoffs()."""

    name: str = "european_call"

    # width of the payoff-smoothing kernel, as a fraction of the terminal
    # value's dispersion; a hard kink would make every second derivative
    # either zero or a delta
    smooth_fraction: float = 0.05


@dataclass(frozen=True)
class SimulationConfig:
    """
    Monte Carlo settings for the training labels, shared by every model.

    `shared_label_keys` reproduces the old behaviour of pricing the whole
    dataset with one random stream. It is kept only so the bias it causes
    stays reproducible and testable - it measurably shifts every label the
    same way and that error never averages out. Leave it False.
    """

    # Exact terminal sampling made a label ~60x cheaper than the old
    # 50-step Euler bundle, so the budget buys far more paths instead.
    num_paths: int = 200_000

    # only read by path-dependent payoffs; a terminal-only payoff is drawn
    # exactly in one step
    num_steps: int = 50

    label_seed: int = 0
    antithetic: bool = True
    shared_label_keys: bool = False

    # paths for the independent benchmark; exact sampling makes a much
    # tighter reference affordable than the labels themselves use
    reference_paths: int = 1_000_000


@dataclass(frozen=True)
class BasketConfig:
    """
    Read only when pricing_model is BASKET_BLACK_SCHOLES.

    All assets share the calibrated single-name volatility and rate, and
    the correlation is an assumption rather than a fit - the option chain
    holds no basket instrument that could determine it. See
    `calibrate_basket`.
    """

    n_assets: int = 3
    correlation: float = 0.5

    # None means equal weights
    weights: Optional[Tuple[float, ...]] = None

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
    # The risk stage stops at the same floor rather than extrapolating.
    min_maturity: float = 0.05

    # Shape of the sampling domain. Spots span a `domain_n_sigma` lognormal
    # move over `domain_horizon` years, matching the exposure simulation;
    # `r_spread` is the band around the calibrated rate, and is only read
    # by models that carry the rate as a feature.
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

    # cheap now that terminal sampling is exact; nine points were far too
    # few for any of the diagnostics to be stable
    n_points: int = 128

    # unrelated to the label seed, so the benchmark is genuinely independent
    seed: int = 12345

    # a breach smaller than this many Monte Carlo standard errors is noise,
    # not a violation
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

        # Cosine decay to a small floor. At a constant rate the loss
        # oscillates by a factor of three for hundreds of epochs and the
        # selected model is whichever epoch was luckiest.
        lr_schedule="cosine",
        lr_final_fraction=0.02,
        warmup_epochs=10,
        gradient_clip=1.0,

        # Selection on the whole objective. Selecting on price+gradient
        # stopped while the curvature term was still improving and left
        # its error 2.6x above what the same run reached later.
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
        """
        Everything a run needs to be reproducible, for config.json.

        `problem` contributes the feature layout, the sampled domain and
        the calibration outcome, which are derived from the config rather
        than stated in it.
        """

        data = asdict(self)

        if problem is not None:
            data["derived"] = problem.describe()

        return data
