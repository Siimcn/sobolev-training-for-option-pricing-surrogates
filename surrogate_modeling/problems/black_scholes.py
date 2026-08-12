import jax
import jax.numpy as jnp
import optimistix as optx
from typing import Tuple
from kalibrierung.calibrator import Calibrator
from marktsimulation.basket_mc import (
    generate_basket_training_paths,
    is_exchangeable,
    simulate_basket_assets,
    uniform_correlation,
)
from marktsimulation.black_scholes import (
    black_scholes_price,
    black_scholes_price_single,
)
from marktsimulation.black_scholes_mc import (
    generate_training_paths,
    make_mc_calibration_pricer,
)
from marktsimulation.mc_pricing import make_feature_price, mc_price
from marktsimulation.payoff import payoff_spec
from marktsimulation.pricing_model import (
    BasketBlackScholesModel,
    BasketBlackScholesParams,
    BlackScholesModel,
    BlackScholesParams,
)
from marktsimulation.timesteppingscheme import EulerMaruyama
from surrogate_modeling.domain import (
    exposure_time_grid,
    lognormal_spot_range,
    maturity_range,
    strike_range,
    uniform,
)
from surrogate_modeling.pricing_problem import (
    CalibrationResult,
    calibration_residuals,
    MonteCarloProblem,
    ProblemSpec,
    ShapeConstraint,
    register_problem,
)

BLACK_SCHOLES = "black_scholes"

BASKET_BLACK_SCHOLES = "basket_black_scholes"


def calibrate_black_scholes(config, market_data) -> CalibrationResult:
    """Fit (r, sigma) to the option chain."""

    if config.market.black_scholes_analytic:
        print("Pricing engine: analytic Black-Scholes")

        def pricing_fn(params, strikes, maturities, is_call):
            return black_scholes_price(
                params=params,
                strikes=strikes,
                maturities=maturities,
                is_call=is_call,
                spot=market_data.spot,
            )

    else:
        print(
            f"Pricing engine: Monte Carlo "
            f"({config.market.mc_calibration_paths} paths, "
            f"{config.market.mc_calibration_steps} steps, "
            f"seed {config.market.mc_calibration_seed})"
        )

        pricing_fn = make_mc_calibration_pricer(
            spot=market_data.spot,
            maturities=market_data.maturities,
            seed=config.market.mc_calibration_seed,
            num_paths=config.market.mc_calibration_paths,
            num_steps=config.market.mc_calibration_steps,
        )

    fitted, solution = Calibrator(pricing_fn=pricing_fn).calibrate(
        BlackScholesParams(
            r=config.market.initial_rate, sigma=config.market.initial_sigma
        ),
        market_data,
    )

    converged = solution.result == optx.RESULTS.successful

    if not converged:
        print(
            f"WARNING: calibration did not converge ({solution.result}). "
            f"Parameters below are the last iterate, not a solution."
        )

    return CalibrationResult(
        params=fitted,
        converged=converged,
        diagnostics={
            "engine": (
                "analytic" if config.market.black_scholes_analytic else "monte_carlo"
            ),
            **calibration_residuals(pricing_fn, fitted, market_data),
        },
    )


def calibrate_basket(config, market_data) -> CalibrationResult:
    """
    Fit (r, sigma) exactly as the single-asset model does, and declare the
    correlation as an assumption.
    """

    result = calibrate_black_scholes(config, market_data)

    print(
        f"Correlation rho = {config.basket.correlation} is assumed, not "
        f"calibrated: the market data contains no basket instrument."
    )

    return CalibrationResult(
        params=result.params,
        converged=result.converged,
        diagnostics=result.diagnostics,
        assumptions={
            "correlation": float(config.basket.correlation),
            "correlation_source": (
                "assumed - the option chain is single-name, so no basket "
                "instrument constrains it"
            ),
            "per_asset_volatility": (
                "all assets share the calibrated single-name sigma"
            ),
        },
    )


class BlackScholesProblem(MonteCarloProblem):
    """Single-asset European option under Black-Scholes."""

    name = BLACK_SCHOLES

    def __init__(self, market_data, calibration, config):
        self.market_data = market_data
        self.calibration = calibration
        self.config = config
        self.params = calibration.params
        self.payoff = config.payoff
        self.simulation = config.simulation
        self.data = config.data
        self.model = BlackScholesModel(scheme=EulerMaruyama())

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return ("S", "K", "T", "sigma", "r")

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        return (
            "Spot Price S",
            "Strike K",
            "Maturity T",
            "Volatility σ",
            "Interest Rate r",
        )

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        spot_low, spot_high = lognormal_spot_range(
            self.market_data,
            self.params.sigma,
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u.at[:, 0].set(uniform(u[:, 0], spot_low, spot_high))

        X = X.at[:, 1].set(uniform(u[:, 1], *strike_range(self.market_data)))

        X = X.at[:, 2].set(
            uniform(u[:, 2], *maturity_range(self.market_data, self.data.min_maturity))
        )

        X = X.at[:, 3].set(
            uniform(u[:, 3], 0.8 * self.params.sigma, 1.2 * self.params.sigma)
        )

        return X.at[:, 4].set(
            uniform(
                u[:, 4],
                self.params.r - self.data.r_spread,
                self.params.r + self.data.r_spread,
            )
        )

    def _price(self, x, key, num_paths):
        return mc_price(
            self.model,
            BlackScholesParams(r=x[4], sigma=x[3]),
            jnp.array([x[0]]),
            x[1],
            x[2],
            key,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.simulation.num_steps,
            smooth_fraction=self.payoff.smooth_fraction,
            antithetic=self.simulation.antithetic,
        )

    def baseline_features(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.market_data.spot,
                float(jnp.median(self.market_data.strikes)),
                float(jnp.median(self.market_data.maturities)),
                self.params.sigma,
                self.params.r,
            ]
        )

    def underlying_paths(self, x: jnp.ndarray, num_paths: int = 100):
        return generate_training_paths(x, num_paths=num_paths)

    def analytic_price(self, x: jnp.ndarray) -> jnp.ndarray:
        return black_scholes_price_single(
            spot=x[0],
            strike=x[1],
            maturity=jnp.maximum(x[2], 1e-8),
            sigma=x[3],
            r=x[4],
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    def arbitrage_bounds(self, x: jnp.ndarray):
        bounds = payoff_spec(self.payoff.name).bounds

        if bounds is None:
            return None

        return bounds(x[0], x[1], jnp.exp(-x[4] * x[2]))

    def shape_constraints(self) -> Tuple[ShapeConstraint, ...]:
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent:
            return ()

        if spec.is_call:
            return (
                ShapeConstraint("S", 0.0, 1.0, "call delta lies in [0, 1]"),
                ShapeConstraint("K", None, 0.0, "a call is cheaper at a higher strike"),
                ShapeConstraint("T", 0.0, None, "more time cannot be worth less"),
                ShapeConstraint("sigma", 0.0, None, "vega is positive"),
                ShapeConstraint("r", 0.0, None, "a call gains from a higher rate"),
            )

        return (
            ShapeConstraint("S", -1.0, 0.0, "put delta lies in [-1, 0]"),
            ShapeConstraint("K", 0.0, None, "a put is worth more at a higher strike"),
            ShapeConstraint("sigma", 0.0, None, "vega is positive"),
        )

    def exposure_paths(
        self,
        strike: float,
        horizon: float = 1.0,
        num_paths: int = 100,
        num_steps: int = 252,
        seed: int = 0,
        min_maturity: float = 0.0,
    ):
        scheme = EulerMaruyama()
        model = BlackScholesModel(scheme=scheme)

        paths = scheme.generate_paths(
            s0=jnp.array([self.market_data.spot]),
            drift_fn=model.drift,
            diffusion_fn=model.diffusion,
            params=self.params,
            key=jax.random.PRNGKey(seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=horizon / num_steps,
        )

        time_grid, remaining = exposure_time_grid(horizon, num_steps, min_maturity)

        spot_paths = paths[:, :, 0]

        features = jnp.stack(
            [
                spot_paths,
                strike * jnp.ones_like(spot_paths),
                jnp.broadcast_to(remaining, spot_paths.shape),
                self.params.sigma * jnp.ones_like(spot_paths),
                self.params.r * jnp.ones_like(spot_paths),
            ],
            axis=-1,
        )

        return time_grid, features

    def describe(self):
        return {
            **super().describe(),
            "payoff": self.payoff.name,
            "calibration": {
                "sigma": float(self.params.sigma),
                "r": float(self.params.r),
                "converged": self.calibration.converged,
                **self.calibration.diagnostics,
            },
            "assumptions": self.calibration.assumptions,
        }


class BasketBlackScholesProblem(MonteCarloProblem):
    """Basket option on `n_assets` correlated Black-Scholes assets."""

    name = BASKET_BLACK_SCHOLES

    def __init__(self, market_data, calibration, config):
        self.market_data = market_data
        self.calibration = calibration
        self.config = config
        self.params = calibration.params
        self.payoff = config.payoff
        self.simulation = config.simulation
        self.data = config.data
        self.basket = config.basket

        self.n_assets = config.basket.n_assets

        self.weights = (
            jnp.full(config.basket.n_assets, 1.0 / config.basket.n_assets)
            if config.basket.weights is None
            else jnp.asarray(config.basket.weights)
        )

        self.corr = uniform_correlation(
            config.basket.n_assets, config.basket.correlation
        )
        self.sigmas = jnp.full(config.basket.n_assets, self.params.sigma)

        if config.basket.symmetrize and not is_exchangeable(
            self.weights, self.sigmas, self.corr
        ):
            raise ValueError(
                "symmetrize requires an exchangeable basket: equal weights, "
                "equal volatilities and a uniform correlation."
            )

        self.model = BasketBlackScholesModel(scheme=EulerMaruyama())

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(f"S{i + 1}" for i in range(self.n_assets)) + ("K", "T")

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        return tuple(f"Spot S{i + 1}" for i in range(self.n_assets)) + (
            "Strike K",
            "Maturity T",
        )

    @property
    def exchangeable_features(self) -> Tuple[int, ...]:
        if not is_exchangeable(self.weights, self.sigmas, self.corr):
            return ()

        return tuple(range(self.n_assets))

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        spot_low, spot_high = lognormal_spot_range(
            self.market_data,
            self.params.sigma,
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u

        for i in range(self.n_assets):
            X = X.at[:, i].set(uniform(u[:, i], spot_low, spot_high))

        X = X.at[:, self.n_assets].set(
            uniform(u[:, self.n_assets], *strike_range(self.market_data))
        )

        return X.at[:, self.n_assets + 1].set(
            uniform(
                u[:, self.n_assets + 1],
                *maturity_range(self.market_data, self.data.min_maturity),
            )
        )

    def _price(self, x, key, num_paths):
        return make_feature_price(
            self.model,
            BasketBlackScholesParams(
                r=self.params.r,
                sigmas=self.sigmas,
                weights=self.weights,
                corr=self.corr,
            ),
            n_assets=self.n_assets,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.simulation.num_steps,
            smooth_fraction=self.payoff.smooth_fraction,
            symmetrize=self.basket.symmetrize,
            antithetic=self.simulation.antithetic,
        )(x, key)

    def baseline_features(self) -> jnp.ndarray:
        return jnp.concatenate(
            [
                jnp.full(self.n_assets, float(self.market_data.spot)),
                jnp.array(
                    [
                        float(jnp.median(self.market_data.strikes)),
                        float(jnp.median(self.market_data.maturities)),
                    ]
                ),
            ]
        )

    def underlying_paths(self, x: jnp.ndarray, num_paths: int = 100):
        return generate_basket_training_paths(
            x,
            weights=self.weights,
            corr=self.corr,
            sigmas=self.sigmas,
            r=self.params.r,
            num_paths=num_paths,
            num_steps=self.simulation.num_steps,
            seed=self.simulation.label_seed,
        )

    def arbitrage_bounds(self, x: jnp.ndarray):
        bounds = payoff_spec(self.payoff.name).bounds

        if bounds is None:
            return None

        basket = jnp.sum(self.weights * x[: self.n_assets])

        return bounds(
            basket, x[self.n_assets], jnp.exp(-self.params.r * x[self.n_assets + 1])
        )

    def shape_constraints(self) -> Tuple[ShapeConstraint, ...]:
        spec = payoff_spec(self.payoff.name)

        if not spec.is_call:
            return ()

        per_asset = float(jnp.max(self.weights))

        constraints = tuple(
            ShapeConstraint(
                f"S{i + 1}", 0.0, per_asset, "a basket delta is bounded by its weight"
            )
            for i in range(self.n_assets)
        )

        if spec.path_dependent:
            return constraints

        return constraints + (
            ShapeConstraint("K", None, 0.0, "a call is cheaper at a higher strike"),
            ShapeConstraint("T", 0.0, None, "more time cannot be worth less"),
        )

    def comonotonic_limit_price(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The price the basket collapses onto when rho = 1 and all spots agree: a
        vanilla option on that common spot.
        """

        return black_scholes_price_single(
            spot=jnp.sum(self.weights * x[: self.n_assets]),
            strike=x[self.n_assets],
            maturity=jnp.maximum(x[self.n_assets + 1], 1e-8),
            sigma=self.params.sigma,
            r=self.params.r,
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    def exposure_paths(
        self,
        strike: float,
        horizon: float = 1.0,
        num_paths: int = 100,
        num_steps: int = 252,
        seed: int = 0,
        min_maturity: float = 0.0,
    ):
        _, paths = simulate_basket_assets(
            s0=jnp.full(self.n_assets, float(self.market_data.spot)),
            weights=self.weights,
            corr=self.corr,
            sigmas=self.sigmas,
            r=self.params.r,
            horizon=horizon,
            num_paths=num_paths,
            num_steps=num_steps,
            key=jax.random.PRNGKey(seed),
        )

        time_grid, remaining = exposure_time_grid(horizon, num_steps, min_maturity)

        scalar_shape = paths.shape[:2]

        features = jnp.concatenate(
            [
                paths,
                jnp.full(scalar_shape + (1,), strike),
                jnp.broadcast_to(remaining, scalar_shape)[..., None],
            ],
            axis=-1,
        )

        return time_grid, features

    def describe(self):
        return {
            **super().describe(),
            "payoff": self.payoff.name,
            "weights": [float(w) for w in self.weights],
            "calibration": {
                "sigma": float(self.params.sigma),
                "r": float(self.params.r),
                "converged": self.calibration.converged,
                **self.calibration.diagnostics,
            },
            "assumptions": self.calibration.assumptions,
        }


register_problem(
    ProblemSpec(BLACK_SCHOLES, BlackScholesProblem, calibrate_black_scholes)
)
register_problem(
    ProblemSpec(BASKET_BLACK_SCHOLES, BasketBlackScholesProblem, calibrate_basket)
)
