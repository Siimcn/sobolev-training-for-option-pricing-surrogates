import jax
import jax.numpy as jnp

from typing import Callable, Dict, Optional, Tuple

from marktsimulation.basket_mc import (
    basket_feature_price,
    generate_basket_training_paths,
    is_exchangeable,
    make_basket_feature_price,
    simulate_basket_assets,
    uniform_correlation,
)

from marktsimulation.black_scholes import black_scholes_price_single

from marktsimulation.black_scholes_mc import (
    bs_mc_feature_price,
    bs_mc_price,
    generate_training_paths,
)

from marktsimulation.pricing_model import (
    BlackScholesModel,
    BlackScholesParams,
)

from marktsimulation.timesteppingscheme import EulerMaruyama

from surrogate_modeling.pricing_problem import (
    PricingProblem,
    register_problem,
)


BLACK_SCHOLES = "black_scholes"
BASKET_BLACK_SCHOLES = "basket_black_scholes"


def _uniform(u, low, high):
    return low + (high - low) * u


def _spot_range(market_data, fitted_params, horizon, n_sigma):
    """
    Spot range covering an `n_sigma` lognormal move over `horizon` years,
    matching the exposure simulation.
    """

    log_spread = n_sigma * fitted_params.sigma * jnp.sqrt(horizon)

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

    return {
        "ITM": 0.85 * spot,
        "ATM": float(spot),
        "OTM": 1.15 * spot,
    }


class BlackScholesProblem(PricingProblem):
    """
    Single-asset European call under Black-Scholes.

    x = [S, K, T, sigma, r] - volatility and rate are features here, so
    the surrogate carries a vega and a rho.
    """

    name = BLACK_SCHOLES

    def __init__(
        self,
        market_data,
        fitted_params,
        min_maturity: Optional[float] = None,
        r_spread: float = 0.02,
        domain_n_sigma: float = 3.0,
        domain_horizon: float = 1.0,
    ):
        self.market_data = market_data
        self.fitted_params = fitted_params
        self.min_maturity = min_maturity
        self.r_spread = r_spread
        self.domain_n_sigma = domain_n_sigma
        self.domain_horizon = domain_horizon

    @property
    def discount_rate(self) -> float:
        return float(self.fitted_params.r)

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
        spot_low, spot_high = _spot_range(
            self.market_data,
            self.fitted_params,
            self.domain_horizon,
            self.domain_n_sigma,
        )

        X = u.at[:, 0].set(_uniform(u[:, 0], spot_low, spot_high))

        X = X.at[:, 1].set(_uniform(u[:, 1], *_strike_range(self.market_data)))

        X = X.at[:, 2].set(
            _uniform(u[:, 2], *_maturity_range(self.market_data, self.min_maturity))
        )

        X = X.at[:, 3].set(
            _uniform(
                u[:, 3],
                0.8 * self.fitted_params.sigma,
                1.2 * self.fitted_params.sigma,
            )
        )

        # r varies around the calibrated market rate, not a fixed constant,
        # so the network sees a training signal for d(price)/dr
        return X.at[:, 4].set(
            _uniform(
                u[:, 4],
                self.fitted_params.r - self.r_spread,
                self.fitted_params.r + self.r_spread,
            )
        )

    def label_price_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return bs_mc_feature_price

    def baseline_features(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.market_data.spot,
                float(jnp.median(self.market_data.strikes)),
                float(jnp.median(self.market_data.maturities)),
                self.fitted_params.sigma,
                self.fitted_params.r,
            ]
        )

    def underlying_paths(self, x: jnp.ndarray, num_paths: int = 100):
        return generate_training_paths(x, num_paths=num_paths)

    def reference_price(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        return bs_mc_price(x, key=key)

    def analytic_price(self, x: jnp.ndarray) -> jnp.ndarray:
        return black_scholes_price_single(
            spot=x[0],
            strike=x[1],
            maturity=jnp.maximum(x[2], 1e-8),
            sigma=x[3],
            r=x[4],
            is_call=True,
        )

    def arbitrage_bounds(self, x: jnp.ndarray) -> Tuple[float, float]:
        spot, strike, maturity, rate = x[0], x[1], x[2], x[4]

        lower = jnp.maximum(spot - strike * jnp.exp(-rate * maturity), 0.0)

        return float(lower), float(spot)

    def exposure_strikes(self) -> Dict[str, float]:
        return _moneyness_strikes(float(self.market_data.spot))

    def exposure_paths(
        self,
        strike: float,
        horizon: float = 1.0,
        num_paths: int = 100,
        num_steps: int = 252,
        seed: int = 0,
    ):
        scheme = EulerMaruyama()
        model = BlackScholesModel(scheme=scheme)

        params = BlackScholesParams(
            r=self.fitted_params.r,
            sigma=self.fitted_params.sigma,
        )

        paths = scheme.generate_paths(
            s0=jnp.array([self.market_data.spot]),
            drift_fn=model.drift,
            diffusion_fn=model.diffusion,
            params=params,
            key=jax.random.PRNGKey(seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=horizon / num_steps,
        )

        time_grid = jnp.linspace(0.0, horizon, num_steps + 1)

        spot_paths = paths[:, :, 0]

        remaining = jnp.maximum(horizon - time_grid, 1e-8)

        features = jnp.stack(
            [
                spot_paths,
                strike * jnp.ones_like(spot_paths),
                jnp.broadcast_to(remaining, spot_paths.shape),
                self.fitted_params.sigma * jnp.ones_like(spot_paths),
                self.fitted_params.r * jnp.ones_like(spot_paths),
            ],
            axis=-1,
        )

        return time_grid, features


class BasketBlackScholesProblem(PricingProblem):
    """
    Arithmetic-average basket call on `n_assets` correlated Black-Scholes
    assets.

    x = [S_1, ..., S_n, K, T]. The basket structure - weights, correlation,
    per-asset vols and the rate - is fixed at construction rather than
    carried in x, so the surrogate has per-asset deltas and gammas but no
    vega or rho.
    """

    name = BASKET_BLACK_SCHOLES

    def __init__(
        self,
        market_data,
        fitted_params,
        n_assets: int = 3,
        correlation: float = 0.5,
        weights: Optional[Tuple[float, ...]] = None,
        num_paths: int = 50_000,
        num_steps: int = 50,
        label_seed: int = 0,
        symmetrize: bool = True,
        min_maturity: Optional[float] = None,
        domain_n_sigma: float = 3.0,
        domain_horizon: float = 1.0,
    ):
        self.market_data = market_data
        self.fitted_params = fitted_params
        self.n_assets = n_assets
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.label_seed = label_seed
        self.symmetrize = symmetrize
        self.min_maturity = min_maturity
        self.domain_n_sigma = domain_n_sigma
        self.domain_horizon = domain_horizon

        self.weights = (
            jnp.full(n_assets, 1.0 / n_assets)
            if weights is None
            else jnp.asarray(weights)
        )

        self.corr = uniform_correlation(n_assets, correlation)
        self.sigmas = jnp.full(n_assets, fitted_params.sigma)

    @property
    def discount_rate(self) -> float:
        return float(self.fitted_params.r)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        spots = tuple(f"S{i + 1}" for i in range(self.n_assets))

        return spots + ("K", "T")

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        spots = tuple(f"Spot S{i + 1}" for i in range(self.n_assets))

        return spots + ("Strike K", "Maturity T")

    @property
    def exchangeable_features(self) -> Tuple[int, ...]:
        if not is_exchangeable(self.weights, self.sigmas, self.corr):
            return ()

        return tuple(range(self.n_assets))

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        spot_low, spot_high = _spot_range(
            self.market_data,
            self.fitted_params,
            self.domain_horizon,
            self.domain_n_sigma,
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
                *_maturity_range(self.market_data, self.min_maturity),
            )
        )

    def label_price_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return make_basket_feature_price(
            weights=self.weights,
            corr=self.corr,
            sigmas=self.sigmas,
            r=self.fitted_params.r,
            num_paths=self.num_paths,
            num_steps=self.num_steps,
            seed=self.label_seed,
            symmetrize=self.symmetrize,
        )

    def baseline_features(self) -> jnp.ndarray:
        spots = jnp.full(self.n_assets, float(self.market_data.spot))

        return jnp.concatenate(
            [
                spots,
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
            r=self.fitted_params.r,
            num_paths=num_paths,
            num_steps=self.num_steps,
            seed=self.label_seed,
        )

    def reference_price(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        return basket_feature_price(
            x,
            weights=self.weights,
            corr=self.corr,
            sigmas=self.sigmas,
            r=self.fitted_params.r,
            key=key,
            num_paths=self.num_paths,
            num_steps=self.num_steps,
            symmetrize=self.symmetrize,
        )

    def arbitrage_bounds(self, x: jnp.ndarray) -> Tuple[float, float]:
        basket = jnp.sum(self.weights * x[: self.n_assets])

        strike = x[self.n_assets]
        maturity = x[self.n_assets + 1]

        discounted = strike * jnp.exp(-self.fitted_params.r * maturity)

        return float(jnp.maximum(basket - discounted, 0.0)), float(basket)

    def exposure_strikes(self) -> Dict[str, float]:
        return _moneyness_strikes(float(self.market_data.spot))

    def exposure_paths(
        self,
        strike: float,
        horizon: float = 1.0,
        num_paths: int = 100,
        num_steps: int = 252,
        seed: int = 0,
    ):
        time_grid, paths = simulate_basket_assets(
            s0=jnp.full(self.n_assets, float(self.market_data.spot)),
            weights=self.weights,
            corr=self.corr,
            sigmas=self.sigmas,
            r=self.fitted_params.r,
            horizon=horizon,
            num_paths=num_paths,
            num_steps=num_steps,
            key=jax.random.PRNGKey(seed),
        )

        remaining = jnp.maximum(horizon - time_grid, 1e-8)

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


def _build_black_scholes(config, market_data, fitted_params) -> PricingProblem:
    return BlackScholesProblem(
        market_data=market_data,
        fitted_params=fitted_params,
        min_maturity=config.data.min_maturity,
        r_spread=config.data.r_spread,
        domain_n_sigma=config.data.domain_n_sigma,
        domain_horizon=config.data.domain_horizon,
    )


def _build_basket_black_scholes(config, market_data, fitted_params) -> PricingProblem:
    return BasketBlackScholesProblem(
        market_data=market_data,
        fitted_params=fitted_params,
        n_assets=config.basket.n_assets,
        correlation=config.basket.correlation,
        weights=config.basket.weights,
        num_paths=config.basket.num_paths,
        num_steps=config.basket.num_steps,
        label_seed=config.basket.label_seed,
        symmetrize=config.basket.symmetrize,
        min_maturity=config.data.min_maturity,
        domain_n_sigma=config.data.domain_n_sigma,
        domain_horizon=config.data.domain_horizon,
    )


register_problem(BLACK_SCHOLES, _build_black_scholes)
register_problem(BASKET_BLACK_SCHOLES, _build_basket_black_scholes)
