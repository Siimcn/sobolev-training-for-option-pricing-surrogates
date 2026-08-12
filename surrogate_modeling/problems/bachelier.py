import jax
import jax.numpy as jnp
import optimistix as optx
from typing import Tuple
from calibration.calibrator import Calibrator
from market_simulation.basket_mc import is_exchangeable, uniform_correlation
from market_simulation.bachelier import (
    bachelier_forward,
    bachelier_price,
    bachelier_price_single,
    bachelier_spot_price,
    basket_bachelier_spot_price,
    basket_normal_volatility,
)
from market_simulation.mc_pricing import make_feature_price, mc_price
from market_simulation.payoff import payoff_spec
from market_simulation.pricing_model import (
    BachelierModel,
    BachelierParams,
    BasketBachelierModel,
    BasketBachelierParams,
)
from market_simulation.timesteppingscheme import EulerMaruyama
from surrogate_modeling.domain import (
    exposure_time_grid,
    maturity_range,
    normal_spot_range,
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

BACHELIER = "bachelier"

BASKET_BACHELIER = "basket_bachelier"


def calibrate_bachelier(config, market_data) -> CalibrationResult:
    """Fit the normal volatility and the rate to the same option chain."""

    def pricing_fn(params, strikes, maturities, is_call):
        return bachelier_price(params, strikes, maturities, is_call, market_data.spot)

    print("Pricing engine: analytic Bachelier (normal model)")

    fitted, solution = Calibrator(pricing_fn=pricing_fn).calibrate(
        BachelierParams(
            sigma=config.market.initial_sigma * float(market_data.spot),
            r=config.market.initial_rate,
        ),
        market_data,
    )

    converged = solution.result == optx.RESULTS.successful

    if not converged:
        print(
            f"WARNING: calibration did not converge ({solution.result}). "
            f"Parameters below are the last iterate, not a solution."
        )

    print(
        f"Normal volatility {float(fitted.sigma):.4f} in price units "
        f"(= {100 * float(fitted.sigma) / float(market_data.spot):.2f}% of spot)"
    )

    return CalibrationResult(
        params=fitted,
        converged=converged,
        diagnostics={
            "engine": "analytic_bachelier",
            "normal_vol_as_pct_of_spot": (
                100 * float(fitted.sigma) / float(market_data.spot)
            ),
            **calibration_residuals(pricing_fn, fitted, market_data),
        },
        assumptions={
            "state": (
                "the modelled state is a forward; the spot is used in its "
                "place, which ignores carry to expiry"
            )
        },
    )


def calibrate_basket_bachelier(config, market_data) -> CalibrationResult:
    """Normal volatility and rate as above; correlation stays an assumption."""

    result = calibrate_bachelier(config, market_data)

    print(
        f"Correlation rho = {config.basket.correlation} is assumed, not "
        f"calibrated: the market data contains no basket instrument."
    )

    return CalibrationResult(
        params=result.params,
        converged=result.converged,
        diagnostics=result.diagnostics,
        assumptions={
            **result.assumptions,
            "correlation": float(config.basket.correlation),
            "correlation_source": (
                "assumed - the option chain is single-name, so no basket "
                "instrument constrains it"
            ),
            "per_asset_volatility": (
                "all assets share the calibrated single-name normal vol"
            ),
        },
    )


def _bachelier_upper_bound(forward, strike, maturity, sigma, discount, is_call):
    """
    A normal underlying is unbounded below, so the Black-Scholes bound "a call
    is worth at most the underlying" does not carry over.
    """

    intrinsic = jnp.maximum((forward - strike) if is_call else (strike - forward), 0.0)

    return float(
        discount
        * (
            intrinsic
            + sigma * jnp.sqrt(jnp.maximum(maturity, 0.0)) * 0.3989422804014327
        )
    )


class BachelierProblem(MonteCarloProblem):
    """Single-asset European option under the Bachelier (normal) model."""

    name = "bachelier"

    def __init__(self, market_data, calibration, config):
        self.market_data = market_data
        self.calibration = calibration
        self.config = config
        self.params = calibration.params
        self.payoff = config.payoff
        self.simulation = config.simulation
        self.data = config.data

        self.model = BachelierModel(scheme=EulerMaruyama())

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return ("S", "K", "T", "sigma", "r")

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        return (
            "Forward S",
            "Strike K",
            "Maturity T",
            "Normal volatility σ",
            "Interest Rate r",
        )

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        low, high = normal_spot_range(
            self.market_data,
            self.params.sigma,
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u.at[:, 0].set(uniform(u[:, 0], low, high))

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

    def _params_at(self, x):
        return BachelierParams(sigma=x[3], r=x[4])

    def _price(self, x, key, num_paths):
        return mc_price(
            self.model,
            self._params_at(x),
            jnp.array([bachelier_forward(x[0], x[2], x[4])]),
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
        num_steps = self.simulation.num_steps

        paths = self.model.scheme.generate_paths(
            s0=jnp.array([x[0]]),
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=BachelierParams(sigma=x[3], r=x[4]),
            key=jax.random.PRNGKey(self.simulation.label_seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=x[2] / num_steps,
        )

        return jnp.linspace(0.0, x[2], num_steps + 1), paths

    def analytic_price(self, x: jnp.ndarray) -> jnp.ndarray:
        return bachelier_spot_price(
            spot=x[0],
            strike=x[1],
            maturity=x[2],
            sigma=x[3],
            r=x[4],
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    def arbitrage_bounds(self, x: jnp.ndarray):
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent:
            return None

        discount = jnp.exp(-x[4] * x[2])

        forward = bachelier_forward(x[0], x[2], x[4])

        intrinsic = jnp.maximum(
            (forward - x[1]) if spec.is_call else (x[1] - forward), 0.0
        )

        return (
            float(discount * intrinsic),
            _bachelier_upper_bound(forward, x[1], x[2], x[3], discount, spec.is_call),
        )

    def shape_constraints(self) -> Tuple[ShapeConstraint, ...]:
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent:
            return ()

        vega = ShapeConstraint("sigma", 0.0, None, "vega is positive")

        if spec.is_call:
            return (
                ShapeConstraint("S", 0.0, 1.0, "call delta lies in [0, 1]"),
                ShapeConstraint("K", None, 0.0, "a call is cheaper at a higher strike"),
                ShapeConstraint("T", 0.0, None, "more time cannot be worth less"),
                vega,
                ShapeConstraint("r", 0.0, None, "a call gains from a higher rate"),
            )

        return (
            ShapeConstraint("S", -1.0, 0.0, "put delta lies in [-1, 0]"),
            ShapeConstraint("K", 0.0, None, "a put is worth more at a higher strike"),
            vega,
        )

    def exposure_paths(
        self,
        strike,
        horizon=1.0,
        num_paths=100,
        num_steps=252,
        seed=0,
        min_maturity=0.0,
    ):
        paths = self.model.scheme.generate_paths(
            s0=jnp.array(
                [bachelier_forward(self.market_data.spot, horizon, self.params.r)]
            ),
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=self.params,
            key=jax.random.PRNGKey(seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=horizon / num_steps,
        )

        time_grid, remaining = exposure_time_grid(horizon, num_steps, min_maturity)

        forward = paths[:, :, 0] * jnp.exp(-self.params.r * remaining)

        features = jnp.stack(
            [
                forward,
                strike * jnp.ones_like(forward),
                jnp.broadcast_to(remaining, forward.shape),
                self.params.sigma * jnp.ones_like(forward),
                self.params.r * jnp.ones_like(forward),
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


class BasketBachelierProblem(MonteCarloProblem):
    """Basket option on correlated driftless normal assets."""

    name = "basket_bachelier"

    def __init__(self, market_data, calibration, config):
        self.market_data = market_data
        self.calibration = calibration
        self.config = config
        self.single_name_params = calibration.params
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
        self.sigmas = jnp.full(config.basket.n_assets, calibration.params.sigma)

        self.model = BasketBachelierModel(scheme=EulerMaruyama())

        self.params = BasketBachelierParams(
            r=calibration.params.r,
            sigmas=self.sigmas,
            weights=self.weights,
            corr=self.corr,
        )

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(f"S{i + 1}" for i in range(self.n_assets)) + ("K", "T")

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        return tuple(f"Forward S{i + 1}" for i in range(self.n_assets)) + (
            "Strike K",
            "Maturity T",
        )

    @property
    def exchangeable_features(self) -> Tuple[int, ...]:
        if not is_exchangeable(self.weights, self.sigmas, self.corr):
            return ()

        return tuple(range(self.n_assets))

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        low, high = normal_spot_range(
            self.market_data,
            self.single_name_params.sigma,
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u

        for i in range(self.n_assets):
            X = X.at[:, i].set(uniform(u[:, i], low, high))

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
            self.params,
            self.n_assets,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.simulation.num_steps,
            smooth_fraction=self.payoff.smooth_fraction,
            symmetrize=self.basket.symmetrize,
            antithetic=self.simulation.antithetic,
            state_fn=lambda spots, maturity: bachelier_forward(
                spots, maturity, self.params.r
            ),
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
        num_steps = self.simulation.num_steps
        maturity = x[self.n_assets + 1]

        paths = self.model.scheme.generate_paths(
            s0=bachelier_forward(x[: self.n_assets], maturity, self.params.r),
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=self.params,
            key=jax.random.PRNGKey(self.simulation.label_seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=maturity / num_steps,
            corr=self.model.noise_correlation(self.params),
        )

        return (
            jnp.linspace(0.0, maturity, num_steps + 1),
            jnp.sum(paths * self.weights, axis=-1),
        )

    def analytic_price(self, x: jnp.ndarray) -> jnp.ndarray:
        """Exact - this is what the normal basket buys over the lognormal one."""

        return basket_bachelier_spot_price(
            spots=x[: self.n_assets],
            strike=x[self.n_assets],
            maturity=x[self.n_assets + 1],
            weights=self.weights,
            sigmas=self.sigmas,
            corr=self.corr,
            r=self.params.r,
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    def comonotonic_limit_price(self, x: jnp.ndarray) -> jnp.ndarray:
        """At rho = 1 the basket volatility is the plain weighted sum."""

        maturity = x[self.n_assets + 1]

        return bachelier_price_single(
            forward=bachelier_forward(
                jnp.sum(self.weights * x[: self.n_assets]), maturity, self.params.r
            ),
            strike=x[self.n_assets],
            maturity=maturity,
            sigma=jnp.sum(self.weights * self.sigmas),
            r=self.params.r,
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    def arbitrage_bounds(self, x: jnp.ndarray):
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent:
            return None

        strike = x[self.n_assets]
        maturity = x[self.n_assets + 1]

        basket = bachelier_forward(
            jnp.sum(self.weights * x[: self.n_assets]), maturity, self.params.r
        )

        discount = jnp.exp(-self.params.r * maturity)

        intrinsic = jnp.maximum(
            (basket - strike) if spec.is_call else (strike - basket), 0.0
        )

        sigma_b = basket_normal_volatility(self.weights, self.sigmas, self.corr)

        return (
            float(discount * intrinsic),
            _bachelier_upper_bound(
                basket, strike, maturity, sigma_b, discount, spec.is_call
            ),
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

    def exposure_paths(
        self,
        strike,
        horizon=1.0,
        num_paths=100,
        num_steps=252,
        seed=0,
        min_maturity=0.0,
    ):
        paths = self.model.scheme.generate_paths(
            s0=bachelier_forward(
                jnp.full(self.n_assets, float(self.market_data.spot)),
                horizon,
                self.params.r,
            ),
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=self.params,
            key=jax.random.PRNGKey(seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=horizon / num_steps,
            corr=self.model.noise_correlation(self.params),
        )

        time_grid, remaining = exposure_time_grid(horizon, num_steps, min_maturity)

        scalar_shape = paths.shape[:2]

        spots = paths * jnp.exp(-self.params.r * remaining)[None, :, None]

        features = jnp.concatenate(
            [
                spots,
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
            "basket_normal_volatility": float(
                basket_normal_volatility(self.weights, self.sigmas, self.corr)
            ),
            "calibration": {
                "sigma": float(self.single_name_params.sigma),
                "r": float(self.single_name_params.r),
                "converged": self.calibration.converged,
                **self.calibration.diagnostics,
            },
            "assumptions": self.calibration.assumptions,
        }


register_problem(ProblemSpec(BACHELIER, BachelierProblem, calibrate_bachelier))
register_problem(
    ProblemSpec(BASKET_BACHELIER, BasketBachelierProblem, calibrate_basket_bachelier)
)
