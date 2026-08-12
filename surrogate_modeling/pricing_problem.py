from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


__all__ = [
    "CalibrationResult",
    "calibration_residuals",
    "PricingProblem",
    "ProblemSpec",
    "SurfaceSpec",
    "available_problems",
    "build_problem",
    "calibrate_problem",
    "problem_spec",
    "register_problem",
]


@dataclass(frozen=True)
class SurfaceSpec:
    """One 2-D slice through the feature space, for a surrogate surface plot."""

    x_index: int
    y_index: int
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]


@dataclass(frozen=True)
class CalibrationResult:
    """What calibration produced, and what it could not."""

    params: Any
    assumptions: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    converged: bool = True


def calibration_residuals(pricing_fn, params, market_data) -> Dict[str, float]:
    """How well the fitted parameters reproduce the instruments they were fitted to."""

    model_prices = pricing_fn(
        params,
        market_data.strikes,
        market_data.maturities,
        market_data.is_call,
    )

    residual = model_prices - market_data.market_prices

    relative = jnp.abs(residual) / jnp.maximum(
        jnp.abs(market_data.market_prices), 1e-8
    )

    return {
        "n_instruments": int(len(market_data.strikes)),
        "residual_rmse": float(jnp.sqrt(jnp.mean(residual**2))),
        "residual_mean_signed": float(jnp.mean(residual)),
        "residual_median_relative_pct": 100.0 * float(jnp.median(relative)),
        "residual_max_absolute": float(jnp.max(jnp.abs(residual))),
    }


class PricingProblem(ABC):
    """What the surrogate is trained to price."""

    name: str = "pricing_problem"

    @property
    @abstractmethod
    def feature_names(self) -> Tuple[str, ...]:
        """Terse per-feature names, used for columns, filenames and printing."""

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        """Spelled-out names for plot axes."""

        return self.feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def discount_rate(self) -> float:
        """Rate the exposure profile is discounted at."""

        return 0.0

    @property
    def exchangeable_features(self) -> Tuple[int, ...]:
        """Feature indices the true price is invariant under permuting."""

        return ()

    @abstractmethod
    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        """Map a uniform (n, n_features) block onto the training domain."""

    @abstractmethod
    def label_price_fn(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """`f(x, key) -> price`, twice differentiable in x."""

    def feature_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Per-feature (low, high) of the training domain."""

        corners = jnp.stack(
            [
                jnp.zeros(self.n_features),
                jnp.ones(self.n_features),
            ]
        )

        bounds = self.sample_features(corners)

        return bounds[0], bounds[1]

    def baseline_features(self) -> jnp.ndarray:
        """A representative in-domain point; the anchor of every surface slice."""

        return self.sample_features(
            jnp.full((1, self.n_features), 0.5)
        )[0]

    def surface_specs(self) -> Tuple[SurfaceSpec, ...]:
        """Which 2-D slices are worth plotting, with ranges from the domain."""

        low, high = self.feature_bounds()

        x_range = (float(low[0]), float(high[0]))

        return tuple(
            SurfaceSpec(
                x_index=0,
                y_index=j,
                x_range=x_range,
                y_range=(float(low[j]), float(high[j])),
            )
            for j in range(1, self.n_features)
        )

    def underlying_paths(
        self,
        x: jnp.ndarray,
        num_paths: int = 100,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        `(time_grid, paths)` of whatever the option is written on, for one
        feature row. None disables the training-path preview.
        """

        return None

    def reference_points(
        self,
        n_points: int = 256,
        seed: int = 12345,
    ) -> jnp.ndarray:
        """In-domain points for the independent benchmark."""

        u = jax.random.uniform(
            jax.random.PRNGKey(seed),
            shape=(n_points, self.n_features),
        )

        return self.sample_features(u)

    def reference_price(
        self,
        x: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Optional[jnp.ndarray]:
        """
        Re-price x by Monte Carlo with an independent `key`, at higher accuracy
        than the training labels.
        """

        return None

    def analytic_price(self, x: jnp.ndarray) -> Optional[jnp.ndarray]:
        """Closed-form price, or None when the model has none."""

        return None

    def arbitrage_bounds(
        self,
        x: jnp.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Model-free (lower, upper) bounds on the price at x."""

        return None

    def comonotonic_limit_price(self, x: jnp.ndarray) -> Optional[jnp.ndarray]:
        """
        The price this model would have if its underlyings were perfectly
        dependent, evaluated at a point where they agree.
        """

        return None

    def shape_constraints(self) -> Tuple["ShapeConstraint", ...]:
        """Sign conditions the true price gradient must satisfy."""

        return ()

    def exposure_paths(
        self,
        strike: float,
        horizon: float = 1.0,
        num_paths: int = 100,
        num_steps: int = 252,
        seed: int = 0,
        min_maturity: float = 0.0,
    ) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        `(time_grid, features)` with features shaped `(num_paths, num_steps +
        1, n_features)`: the surrogate's own inputs along a simulated future,
        for the exposure calculation.
        """

        return None

    def exposure_strikes(self) -> Dict[str, float]:
        """Strikes the exposure profile is reported at, by label."""

        return {}

    def describe(self) -> Dict[str, object]:
        """What this problem contributes to the archived run configuration."""

        low, high = self.feature_bounds()

        return {
            "problem": self.name,
            "feature_names": list(self.feature_names),
            "n_features": self.n_features,
            "domain_low": [float(v) for v in low],
            "domain_high": [float(v) for v in high],
            "exchangeable_features": list(self.exchangeable_features),
        }


@dataclass(frozen=True)
class ShapeConstraint:
    """"d(price)/d(feature) must be within [low, high]"."""

    feature: str
    low: Optional[float] = None
    high: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class ProblemSpec:
    """A registered pricing problem: how to calibrate it, and how to build it."""

    name: str
    build: Callable[..., PricingProblem]
    calibrate: Callable[..., CalibrationResult]


_REGISTRY: Dict[str, ProblemSpec] = {}


def register_problem(spec: ProblemSpec, overwrite: bool = False) -> None:
    """Make `spec.name` selectable as `data.pricing_model`."""

    key = spec.name.lower()

    if key in _REGISTRY and not overwrite:
        raise ValueError(
            f"A pricing problem named '{spec.name}' is already registered. "
            f"Pass overwrite=True to replace it."
        )

    _REGISTRY[key] = spec


def available_problems() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def problem_spec(name: str) -> ProblemSpec:
    key = name.lower()

    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown pricing model '{name}'. "
            f"Expected one of: {', '.join(available_problems())}."
        )

    return _REGISTRY[key]


def calibrate_problem(name: str, config, market_data) -> CalibrationResult:
    """Fit the model behind `name` to the market data it was given."""

    return problem_spec(name).calibrate(config=config, market_data=market_data)


def build_problem(name: str, **kwargs) -> PricingProblem:
    """Construct the registered problem `name` from config and calibration."""

    return problem_spec(name).build(**kwargs)
