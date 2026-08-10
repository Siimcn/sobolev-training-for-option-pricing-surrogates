from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


__all__ = [
    "CalibrationResult",
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
    """
    What calibration produced, and what it could not.

    `assumptions` records parameters the market data does not determine -
    a basket correlation with no basket quotes to fit it to, for instance.
    They are archived alongside the fitted values so a reader can never
    mistake one for the other.
    """

    params: Any
    assumptions: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    converged: bool = True


class PricingProblem(ABC):
    """
    What the surrogate is trained to price.

    Every model-specific fact lives behind this interface: how the model
    is calibrated, the feature layout, the sampling domain, the label
    pricer, which slices are worth plotting, how a trained surrogate is
    independently validated, and how future states are simulated for an
    exposure profile. The pipeline stages take a PricingProblem and never
    ask which model it holds, so adding a model is an implementation plus
    a registration rather than an edit in every stage.

    Only `name`, `feature_names`, `sample_features` and `label_price_fn`
    are required. Everything else is derived from those by default, so a
    new problem is usable before it is complete: the stages it does not
    support switch themselves off instead of producing wrong output.

    One convention, relied on by the default plots and the exposure chart:
    feature 0 is the primary underlying. It is the axis a price surface is
    swept against and the series a path plot shows. A problem whose first
    feature is something else stays correct - the axes are labelled from
    `feature_names` either way - but its default plots will be less
    useful, so override `surface_specs`.
    """

    name: str = "pricing_problem"

    # ---------------------------------------------------------- layout

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
        """
        Feature indices the true price is invariant under permuting.

        Empty unless the model says otherwise; the validation stage turns
        this into a check on the trained surrogate.
        """

        return ()

    # ------------------------------------------------------------ data

    @abstractmethod
    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Map a uniform (n, n_features) block onto the training domain.

        Must be coordinate-wise increasing in `u`, so `feature_bounds` can
        recover the domain from its corners.
        """

    @abstractmethod
    def label_price_fn(self) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        `f(x, key) -> price`, twice differentiable in x.

        Randomness is an argument, not a closure. Every sample is priced
        with its own key so the Monte Carlo error averages out over the
        dataset instead of shifting all of it the same way; see
        `create_sobolev_labels` for the measurement behind that choice.
        """

    def feature_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Per-feature (low, high) of the training domain.

        Read off `sample_features` rather than declared a second time, so
        a plot range and the range the surrogate was trained on cannot
        drift apart.
        """

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

    # --------------------------------------------------------- plotting

    def surface_specs(self) -> Tuple[SurfaceSpec, ...]:
        """
        Which 2-D slices are worth plotting, with ranges from the domain.

        The default sweeps the first feature against every other one.
        """

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

    # ------------------------------------------------------- validation

    def reference_points(
        self,
        n_points: int = 256,
        seed: int = 12345,
    ) -> jnp.ndarray:
        """
        In-domain points for the independent benchmark.

        Drawn from the problem's own domain with a seed unrelated to the
        training draw, so the benchmark covers where the surrogate is
        meant to work rather than where it happened to be fitted.
        """

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
        Re-price x by Monte Carlo with an independent `key`, at higher
        accuracy than the training labels.

        None means the problem offers no independent benchmark, which the
        validation stage reports rather than passes over.
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

        For a multi-asset problem this is the no-diversification limit: a
        basket call must be worth no more than it. It is the one
        consequence of an *assumed* dependence structure that the data can
        still test, which is why the validation stage checks it.

        None where the concept does not apply.
        """

        return None

    def shape_constraints(self) -> Tuple["ShapeConstraint", ...]:
        """
        Sign conditions the true price gradient must satisfy.

        Cheap, model-free, and independent of any reference price, so they
        catch errors the aggregate metrics average away.
        """

        return ()

    # ------------------------------------------------------------ risk

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
        `(time_grid, features)` with features shaped
        `(num_paths, num_steps + 1, n_features)`: the surrogate's own
        inputs along a simulated future, for the exposure calculation.

        `min_maturity` is the floor the training domain used. The profile
        stops there rather than running the remaining maturity to zero,
        which would evaluate the surrogate outside the region it was
        fitted on.

        None disables the risk stage.
        """

        return None

    def exposure_strikes(self) -> Dict[str, float]:
        """Strikes the exposure profile is reported at, by label."""

        return {}

    def describe(self) -> Dict[str, object]:
        """
        What this problem contributes to the archived run configuration.

        Cheap by construction: capability is discovered by calling the
        optional methods and checking for None, never by a declared flag
        that could drift away from what they actually return.
        """

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
    """
    "d(price)/d(feature) must be within [low, high]".

    `low`/`high` are None where the side is unbounded.
    """

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
