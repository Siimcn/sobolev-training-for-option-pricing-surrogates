import jax
import jax.numpy as jnp
import optimistix as optx

from typing import Callable, Dict, Optional, Tuple

from kalibrierung.calibrator import Calibrator

from marktsimulation.basket_mc import (
    basket_feature_price,
    generate_basket_training_paths,
    is_exchangeable,
    make_basket_feature_price,
    simulate_basket_assets,
    uniform_correlation,
)

from marktsimulation.black_scholes import (
    black_scholes_price,
    black_scholes_price_single,
)

from marktsimulation.black_scholes_mc import (
    bs_mc_price,
    generate_training_paths,
    make_mc_calibration_pricer,
)

from marktsimulation.payoff import payoff_spec

from marktsimulation.pricing_model import (
    BlackScholesModel,
    BlackScholesParams,
)

from marktsimulation.timesteppingscheme import EulerMaruyama

from surrogate_modeling.pricing_problem import (
    CalibrationResult,
    PricingProblem,
    ProblemSpec,
    ShapeConstraint,
    register_problem,
)


BLACK_SCHOLES = "black_scholes"
BASKET_BLACK_SCHOLES = "basket_black_scholes"


# ------------------------------------------------------------- calibration


def calibrate_black_scholes(config, market_data) -> CalibrationResult:
    """
    Fit (r, sigma) to the option chain.

    The engine is a choice, not a model property: the closed form is fast
    and bitwise reproducible, the Monte Carlo branch exists for models
    that have none and inherits the simulation's own bias.
    """

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
            r=config.market.initial_rate,
            sigma=config.market.initial_sigma,
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
            "n_instruments": int(len(market_data.strikes)),
            "engine": (
                "analytic" if config.market.black_scholes_analytic else "monte_carlo"
            ),
        },
    )


def calibrate_basket(config, market_data) -> CalibrationResult:
    """
    Fit (r, sigma) exactly as the single-asset model does, and declare the
    correlation as an assumption.

    There is no basket quote in the market data - the chain is single-name
    options on one ticker - so nothing here determines the correlation.
    Fitting it would mean inventing information. It is recorded as an
    assumption instead, and the validation stage checks the one property
    that *is* testable: at rho = 1 the basket must collapse onto the
    single-asset price.
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


# ------------------------------------------------------------- domain help


def _uniform(u, low, high):
    return low + (high - low) * u


def _spot_range(market_data, params, horizon, n_sigma):
    """Spot range covering an `n_sigma` lognormal move over `horizon` years."""

    log_spread = n_sigma * params.sigma * jnp.sqrt(horizon)

    return (
        market_data.spot * jnp.exp(-log_spread),
        market_data.spot * jnp.exp(log_spread),
    )


def _maturity_range(market_data, min_maturity):
    """
    Market maturities, optionally floored. The shortest expiries carry a
    near-discontinuous payoff, so their curvature labels are orders of
    magnitude larger than the rest of the domain.
    """

    low = float(jnp.min(market_data.maturities))
    high = float(jnp.max(market_data.maturities))

    if min_maturity is not None:
        low = max(low, float(min_maturity))

    if low >= high:
        raise ValueError(
            f"min_maturity {min_maturity} leaves no maturity range below "
            f"the longest market expiry {high}."
        )

    return low, high


def _strike_range(market_data):
    return (
        float(jnp.min(market_data.strikes)),
        float(jnp.max(market_data.strikes)),
    )


def _moneyness_strikes(spot: float) -> Dict[str, float]:
    """A deep ITM call is near-linear in S, so ATM catches Greeks errors it hides."""

    return {"ITM": 0.85 * spot, "ATM": float(spot), "OTM": 1.15 * spot}


class BlackScholesProblem(PricingProblem):
    """
    Single-asset European option under Black-Scholes.

    x = [S, K, T, sigma, r] - volatility and rate are features here, so
    the surrogate carries a vega and a rho.
    """

    name = BLACK_SCHOLES

    def __init__(self, market_data, calibration, payoff, simulation, data):
        self.market_data = market_data
        self.calibration = calibration
        self.params = calibration.params
        self.payoff = payoff
        self.simulation = simulation
        self.data = data

    # ------------------------------------------------------------ layout

    @property
    def discount_rate(self) -> float:
        return float(self.params.r)

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

    # -------------------------------------------------------------- data

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        spot_low, spot_high = _spot_range(
            self.market_data,
            self.params,
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u.at[:, 0].set(_uniform(u[:, 0], spot_low, spot_high))

        X = X.at[:, 1].set(_uniform(u[:, 1], *_strike_range(self.market_data)))

        X = X.at[:, 2].set(
            _uniform(
                u[:, 2], *_maturity_range(self.market_data, self.data.min_maturity)
            )
        )

        X = X.at[:, 3].set(
            _uniform(u[:, 3], 0.8 * self.params.sigma, 1.2 * self.params.sigma)
        )

        # r varies around the calibrated market rate, not a fixed constant,
        # so the network sees a training signal for d(price)/dr
        return X.at[:, 4].set(
            _uniform(
                u[:, 4],
                self.params.r - self.data.r_spread,
                self.params.r + self.data.r_spread,
            )
        )

    def _price(self, x, key, num_paths):
        return bs_mc_price(
            x,
            key,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.simulation.num_steps,
            antithetic=self.simulation.antithetic,
        )

    def label_price_fn(self):
        return lambda x, key: self._price(x, key, self.simulation.num_paths)

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

    # -------------------------------------------------------- validation

    def reference_price(self, x, key):
        return self._price(x, key, self.simulation.reference_paths)

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

    # -------------------------------------------------------------- risk

    def exposure_strikes(self) -> Dict[str, float]:
        return _moneyness_strikes(float(self.market_data.spot))

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

        time_grid, remaining = _exposure_time_grid(horizon, num_steps, min_maturity)

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


class BasketBlackScholesProblem(PricingProblem):
    """
    Basket option on `n_assets` correlated Black-Scholes assets.

    x = [S_1, ..., S_n, K, T]. The basket structure - weights,
    correlation, per-asset vols and the rate - is fixed at construction
    rather than carried in the feature vector, so the surrogate has
    per-asset deltas and gammas but no vega or rho.
    """

    name = BASKET_BLACK_SCHOLES

    def __init__(self, market_data, calibration, payoff, simulation, data, basket):
        self.market_data = market_data
        self.calibration = calibration
        self.params = calibration.params
        self.payoff = payoff
        self.simulation = simulation
        self.data = data
        self.basket = basket

        self.n_assets = basket.n_assets

        self.weights = (
            jnp.full(basket.n_assets, 1.0 / basket.n_assets)
            if basket.weights is None
            else jnp.asarray(basket.weights)
        )

        self.corr = uniform_correlation(basket.n_assets, basket.correlation)
        self.sigmas = jnp.full(basket.n_assets, self.params.sigma)

    # ------------------------------------------------------------ layout

    @property
    def discount_rate(self) -> float:
        return float(self.params.r)

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

    # -------------------------------------------------------------- data

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        spot_low, spot_high = _spot_range(
            self.market_data,
            self.params,
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u

        for i in range(self.n_assets):
            X = X.at[:, i].set(_uniform(u[:, i], spot_low, spot_high))

        X = X.at[:, self.n_assets].set(
            _uniform(u[:, self.n_assets], *_strike_range(self.market_data))
        )

        return X.at[:, self.n_assets + 1].set(
            _uniform(
                u[:, self.n_assets + 1],
                *_maturity_range(self.market_data, self.data.min_maturity),
            )
        )

    def _pricer(self, num_paths):
        return make_basket_feature_price(
            weights=self.weights,
            corr=self.corr,
            sigmas=self.sigmas,
            r=self.params.r,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.simulation.num_steps,
            smooth_fraction=self.payoff.smooth_fraction,
            symmetrize=self.basket.symmetrize,
            antithetic=self.simulation.antithetic,
        )

    def label_price_fn(self):
        return self._pricer(self.simulation.num_paths)

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

    # -------------------------------------------------------- validation

    def reference_price(self, x, key):
        return self._pricer(self.simulation.reference_paths)(x, key)

    def arbitrage_bounds(self, x: jnp.ndarray):
        bounds = payoff_spec(self.payoff.name).bounds

        if bounds is None:
            return None

        basket = jnp.sum(self.weights * x[: self.n_assets])

        return bounds(
            basket,
            x[self.n_assets],
            jnp.exp(-self.params.r * x[self.n_assets + 1]),
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
        The price the basket collapses onto when rho = 1 and all spots
        agree: a vanilla option on that common spot.

        Diversification can only reduce a call's value, so the simulated
        basket must not exceed this. It is the one consequence of the
        assumed correlation that the data can still test.
        """

        return black_scholes_price_single(
            spot=jnp.sum(self.weights * x[: self.n_assets]),
            strike=x[self.n_assets],
            maturity=jnp.maximum(x[self.n_assets + 1], 1e-8),
            sigma=self.params.sigma,
            r=self.params.r,
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    # -------------------------------------------------------------- risk

    def exposure_strikes(self) -> Dict[str, float]:
        return _moneyness_strikes(float(self.market_data.spot))

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

        time_grid, remaining = _exposure_time_grid(horizon, num_steps, min_maturity)

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


def _exposure_time_grid(horizon, num_steps, min_maturity):
    """
    Stop the exposure profile at the training floor.

    Running the remaining maturity down to zero evaluates the surrogate
    below `min_maturity`, outside the domain it was fitted on. That is
    where a long call was previously valued negative, which produced a
    non-zero DVA.
    """

    time_grid = jnp.linspace(0.0, horizon, num_steps + 1)

    floor = max(float(min_maturity), 1e-8)

    return time_grid, jnp.maximum(horizon - time_grid, floor)


def _build_black_scholes(config, market_data, calibration) -> PricingProblem:
    return BlackScholesProblem(
        market_data=market_data,
        calibration=calibration,
        payoff=config.payoff,
        simulation=config.simulation,
        data=config.data,
    )


def _build_basket(config, market_data, calibration) -> PricingProblem:
    return BasketBlackScholesProblem(
        market_data=market_data,
        calibration=calibration,
        payoff=config.payoff,
        simulation=config.simulation,
        data=config.data,
        basket=config.basket,
    )


register_problem(ProblemSpec(BLACK_SCHOLES, _build_black_scholes, calibrate_black_scholes))
register_problem(ProblemSpec(BASKET_BLACK_SCHOLES, _build_basket, calibrate_basket))
