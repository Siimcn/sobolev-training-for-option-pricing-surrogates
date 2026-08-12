import jax
import jax.numpy as jnp
import optimistix as optx

from typing import Dict, Tuple

from kalibrierung.calibrator import Calibrator

from marktsimulation.basket_mc import uniform_correlation

from marktsimulation.heston import (
    feller_ratio,
    heston_price,
    heston_price_vector,
)

from marktsimulation.mc_pricing import mc_price

from marktsimulation.payoff import payoff_spec

from marktsimulation.pricing_model import (
    BasketHestonModel,
    BasketHestonParams,
    HestonModel,
    HestonParams,
)

from marktsimulation.timesteppingscheme import EulerMaruyama

from surrogate_modeling.domain import (
    exposure_time_grid,
    lognormal_spot_range,
    maturity_range,
    moneyness_strikes,
    strike_range,
    uniform,
)

from surrogate_modeling.pricing_problem import (
    CalibrationResult,
    calibration_residuals,
    PricingProblem,
    ProblemSpec,
    ShapeConstraint,
    register_problem,
)


HESTON = "heston"
BASKET_HESTON = "basket_heston"


def _heston_transforms():
    """
    Fit in an unconstrained space so the solver cannot leave the admissible
    region: a log for each positive parameter, an arctanh for the correlation.
    """

    def to_model(z):
        return HestonParams(
            r=z.r,
            kappa=jnp.exp(z.kappa),
            theta=jnp.exp(z.theta),
            xi=jnp.exp(z.xi),
            rho=jnp.tanh(z.rho),
            nu0=jnp.exp(z.nu0),
        )

    def to_unconstrained(p):
        return HestonParams(
            r=p.r,
            kappa=jnp.log(p.kappa),
            theta=jnp.log(p.theta),
            xi=jnp.log(p.xi),
            rho=jnp.arctanh(p.rho),
            nu0=jnp.log(p.nu0),
        )

    return to_model, to_unconstrained


def calibrate_heston(config, market_data) -> CalibrationResult:
    """Fit all six parameters to the option chain by Fourier pricing."""

    print("Pricing engine: semi-analytic Heston (Fourier inversion)")

    to_model, to_unconstrained = _heston_transforms()

    def pricing_fn(params, strikes, maturities, is_call):
        return heston_price_vector(
            params, strikes, maturities, is_call, market_data.spot
        )

    variance = float(config.market.initial_sigma) ** 2

    fitted, solution = Calibrator(
        pricing_fn=pricing_fn,
        transform_fn=to_model,
        inv_transform_fn=to_unconstrained,
    ).calibrate(
        HestonParams(
            r=config.market.initial_rate,
            kappa=config.heston.initial_kappa,
            theta=variance,
            xi=config.heston.initial_xi,
            rho=config.heston.initial_rho,
            nu0=variance,
        ),
        market_data,
    )

    converged = solution.result == optx.RESULTS.successful

    if not converged:
        print(
            f"WARNING: calibration did not converge ({solution.result}). "
            f"Parameters below are the last iterate, not a solution."
        )

    ratio = feller_ratio(fitted)

    print(
        f"kappa {float(fitted.kappa):.4f}  theta {float(fitted.theta):.5f}  "
        f"xi {float(fitted.xi):.4f}  rho {float(fitted.rho):+.4f}  "
        f"v0 {float(fitted.nu0):.5f}"
    )
    print(
        f"Feller ratio 2*kappa*theta/xi^2 = {ratio:.3f}"
        + ("" if ratio >= 1.0 else "   (violated: the variance reaches zero)")
    )

    return CalibrationResult(
        params=fitted,
        converged=converged,
        diagnostics={
            "engine": "fourier_heston",
            "feller_ratio": ratio,
            "feller_satisfied": bool(ratio >= 1.0),
            **calibration_residuals(pricing_fn, fitted, market_data),
        },
    )


def calibrate_basket_heston(config, market_data) -> CalibrationResult:
    """Single-name Heston fit; the asset correlation stays an assumption."""

    result = calibrate_heston(config, market_data)

    print(
        f"Asset correlation {config.basket.correlation} is assumed, not "
        f"calibrated: the market data contains no basket instrument."
    )

    return CalibrationResult(
        params=result.params,
        converged=result.converged,
        diagnostics=result.diagnostics,
        assumptions={
            "asset_correlation": float(config.basket.correlation),
            "asset_correlation_source": (
                "assumed - the option chain is single-name, so no basket "
                "instrument constrains it"
            ),
            "per_asset_dynamics": (
                "every asset shares the calibrated single-name Heston "
                "parameters"
            ),
            "spot_variance_cross_correlation": (
                "corr(dW_Si, dW_vj) = rho * C_ij for i != j, from the "
                "Kronecker structure [[1, rho], [rho, 1]] (x) C - that asset "
                "i's spot is correlated with asset j's variance in this way "
                "is a modelling choice, with no market instrument behind it"
            ),
        },
    )


class HestonProblem(PricingProblem):
    """Single asset under Heston stochastic volatility."""

    name = HESTON

    def __init__(self, market_data, calibration, payoff, simulation, data, heston):
        self.market_data = market_data
        self.calibration = calibration
        self.params = calibration.params
        self.payoff = payoff
        self.simulation = simulation
        self.data = data
        self.heston = heston

        self.model = HestonModel(scheme=EulerMaruyama())

    @property
    def discount_rate(self) -> float:
        return float(self.params.r)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return ("S", "K", "T", "v0", "kappa", "theta", "xi", "rho")

    @property
    def feature_labels(self) -> Tuple[str, ...]:
        return (
            "Spot Price S",
            "Strike K",
            "Maturity T",
            "Initial variance v0",
            "Mean reversion kappa",
            "Long-run variance theta",
            "Vol of vol xi",
            "Spot-vol correlation rho",
        )

    def _params_at(self, x):
        return HestonParams(
            r=self.params.r, kappa=x[4], theta=x[5], xi=x[6], rho=x[7], nu0=x[3]
        )

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        band = self.heston.parameter_band

        low, high = lognormal_spot_range(
            self.market_data,
            jnp.sqrt(self.params.theta),
            self.data.domain_horizon,
            self.data.domain_n_sigma,
        )

        X = u.at[:, 0].set(uniform(u[:, 0], low, high))
        X = X.at[:, 1].set(uniform(u[:, 1], *strike_range(self.market_data)))
        X = X.at[:, 2].set(
            uniform(
                u[:, 2], *maturity_range(self.market_data, self.data.min_maturity)
            )
        )

        for index, value in (
            (3, self.params.nu0),
            (4, self.params.kappa),
            (5, self.params.theta),
            (6, self.params.xi),
        ):
            X = X.at[:, index].set(
                uniform(u[:, index], (1.0 - band) * value, (1.0 + band) * value)
            )

        return X.at[:, 7].set(
            uniform(
                u[:, 7],
                max(float(self.params.rho) - band, -0.95),
                min(float(self.params.rho) + band, 0.95),
            )
        )

    def _price(self, x, key, num_paths):
        return mc_price(
            self.model,
            self._params_at(x),
            jnp.array([x[0], x[3]]),
            x[1],
            x[2],
            key,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.heston.num_steps,
            smooth_fraction=self.payoff.smooth_fraction,
            antithetic=self.simulation.antithetic,
            value_fn=lambda state: state[0],
        )

    def label_price_fn(self):
        return lambda x, key: self._price(x, key, self.heston.num_paths)

    def reference_price(self, x, key):
        return self._price(x, key, self.heston.reference_paths)

    def analytic_price(self, x: jnp.ndarray) -> jnp.ndarray:
        """Fourier inversion: exact up to quadrature, and differentiable."""

        return heston_price(
            x[0],
            x[1],
            x[2],
            self._params_at(x),
            is_call=payoff_spec(self.payoff.name).is_call,
        )

    def baseline_features(self) -> jnp.ndarray:
        return jnp.array(
            [
                self.market_data.spot,
                float(jnp.median(self.market_data.strikes)),
                float(jnp.median(self.market_data.maturities)),
                self.params.nu0,
                self.params.kappa,
                self.params.theta,
                self.params.xi,
                self.params.rho,
            ]
        )

    def underlying_paths(self, x: jnp.ndarray, num_paths: int = 100):
        num_steps = self.heston.num_steps

        params = self._params_at(x)

        paths = self.model.scheme.generate_paths(
            s0=jnp.array([x[0], x[3]]),
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=params,
            key=jax.random.PRNGKey(self.simulation.label_seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=x[2] / num_steps,
            corr=self.model.noise_correlation(params),
        )

        return jnp.linspace(0.0, x[2], num_steps + 1), paths[:, :, 0]

    def arbitrage_bounds(self, x: jnp.ndarray):
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent or spec.bounds is None:
            return None

        return spec.bounds(x[0], x[1], jnp.exp(-self.params.r * x[2]))

    def shape_constraints(self) -> Tuple[ShapeConstraint, ...]:
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent or not spec.is_call:
            return ()

        return (
            ShapeConstraint("S", 0.0, 1.0, "call delta lies in [0, 1]"),
            ShapeConstraint("K", None, 0.0, "a call is cheaper at a higher strike"),
            ShapeConstraint("T", 0.0, None, "more time cannot be worth less"),
            ShapeConstraint("v0", 0.0, None, "more variance is worth more"),
            ShapeConstraint(
                "theta", 0.0, None, "a higher long-run variance is worth more"
            ),
        )

    def exposure_strikes(self) -> Dict[str, float]:
        return moneyness_strikes(float(self.market_data.spot))

    def exposure_paths(
        self, strike, horizon=1.0, num_paths=100, num_steps=252, seed=0,
        min_maturity=0.0,
    ):
        paths = self.model.scheme.generate_paths(
            s0=jnp.array([self.market_data.spot, self.params.nu0]),
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

        spot_paths = paths[:, :, 0]
        variance_paths = paths[:, :, 1]

        ones = jnp.ones_like(spot_paths)

        features = jnp.stack(
            [
                spot_paths,
                strike * ones,
                jnp.broadcast_to(remaining, spot_paths.shape),
                variance_paths,
                self.params.kappa * ones,
                self.params.theta * ones,
                self.params.xi * ones,
                self.params.rho * ones,
            ],
            axis=-1,
        )

        return time_grid, features

    def describe(self):
        return {
            **super().describe(),
            "payoff": self.payoff.name,
            "calibration": _calibration_record(self.params, self.calibration),
            "assumptions": self.calibration.assumptions,
        }


class BasketHestonProblem(PricingProblem):
    """
    Basket on correlated Heston assets, every asset sharing the calibrated
    dynamics.
    """

    name = BASKET_HESTON

    def __init__(
        self, market_data, calibration, payoff, simulation, data, basket, heston
    ):
        self.market_data = market_data
        self.calibration = calibration
        self.scalar_params = calibration.params
        self.payoff = payoff
        self.simulation = simulation
        self.data = data
        self.basket = basket
        self.heston = heston

        self.n_assets = basket.n_assets

        self.weights = (
            jnp.full(basket.n_assets, 1.0 / basket.n_assets)
            if basket.weights is None
            else jnp.asarray(basket.weights)
        )

        self.corr = uniform_correlation(basket.n_assets, basket.correlation)

        self.model = BasketHestonModel(scheme=EulerMaruyama())

        self.params = BasketHestonParams(
            r=calibration.params.r,
            kappa=calibration.params.kappa,
            theta=calibration.params.theta,
            xi=calibration.params.xi,
            rho=calibration.params.rho,
            nu0=calibration.params.nu0,
            weights=self.weights,
            corr=self.corr,
        )

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
        equal_weights = bool(jnp.allclose(self.weights, self.weights[0]))

        off_diagonal = self.corr[~jnp.eye(self.n_assets, dtype=bool)]

        uniform_corr = len(off_diagonal) == 0 or bool(
            jnp.allclose(off_diagonal, off_diagonal[0])
        )

        if equal_weights and uniform_corr:
            return tuple(range(self.n_assets))

        return ()

    def sample_features(self, u: jnp.ndarray) -> jnp.ndarray:
        low, high = lognormal_spot_range(
            self.market_data,
            jnp.sqrt(self.scalar_params.theta),
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

    def _initial_state(self, spots):
        """The simulated state is the spots followed by their variances."""

        return jnp.concatenate(
            [spots, jnp.full(self.n_assets, self.scalar_params.nu0)]
        )

    def _price(self, x, key, num_paths):
        spots = (
            jnp.sort(x[: self.n_assets])
            if self.basket.symmetrize
            else x[: self.n_assets]
        )

        return mc_price(
            self.model,
            self.params,
            self._initial_state(spots),
            x[self.n_assets],
            x[self.n_assets + 1],
            key,
            payoff=self.payoff.name,
            num_paths=num_paths,
            num_steps=self.heston.num_steps,
            smooth_fraction=self.payoff.smooth_fraction,
            antithetic=self.simulation.antithetic,
        )

    def label_price_fn(self):
        return lambda x, key: self._price(x, key, self.heston.num_paths)

    def reference_price(self, x, key):
        return self._price(x, key, self.heston.reference_paths)

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
        num_steps = self.heston.num_steps
        maturity = x[self.n_assets + 1]

        paths = self.model.scheme.generate_paths(
            s0=self._initial_state(x[: self.n_assets]),
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=self.params,
            key=jax.random.PRNGKey(self.simulation.label_seed),
            num_paths=num_paths,
            num_steps=num_steps,
            dt=maturity / num_steps,
            corr=self.model.noise_correlation(self.params),
        )

        basket = jnp.sum(paths[:, :, : self.n_assets] * self.weights, axis=-1)

        return jnp.linspace(0.0, maturity, num_steps + 1), basket

    def arbitrage_bounds(self, x: jnp.ndarray):
        spec = payoff_spec(self.payoff.name)

        if spec.path_dependent or spec.bounds is None:
            return None

        basket = jnp.sum(self.weights * x[: self.n_assets])

        return spec.bounds(
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

    def exposure_strikes(self) -> Dict[str, float]:
        return moneyness_strikes(float(self.market_data.spot))

    def exposure_paths(
        self, strike, horizon=1.0, num_paths=100, num_steps=252, seed=0,
        min_maturity=0.0,
    ):
        paths = self.model.scheme.generate_paths(
            s0=self._initial_state(
                jnp.full(self.n_assets, float(self.market_data.spot))
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

        spots = paths[:, :, : self.n_assets]

        scalar_shape = spots.shape[:2]

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
            "calibration": _calibration_record(self.scalar_params, self.calibration),
            "assumptions": self.calibration.assumptions,
        }


def _calibration_record(params, calibration) -> Dict[str, object]:
    return {
        "r": float(params.r),
        "kappa": float(params.kappa),
        "theta": float(params.theta),
        "xi": float(params.xi),
        "rho": float(params.rho),
        "nu0": float(params.nu0),
        "converged": calibration.converged,
        **calibration.diagnostics,
    }


def _build_heston(config, market_data, calibration) -> PricingProblem:
    return HestonProblem(
        market_data=market_data,
        calibration=calibration,
        payoff=config.payoff,
        simulation=config.simulation,
        data=config.data,
        heston=config.heston,
    )


def _build_basket_heston(config, market_data, calibration) -> PricingProblem:
    return BasketHestonProblem(
        market_data=market_data,
        calibration=calibration,
        payoff=config.payoff,
        simulation=config.simulation,
        data=config.data,
        basket=config.basket,
        heston=config.heston,
    )


register_problem(ProblemSpec(HESTON, _build_heston, calibrate_heston))
register_problem(
    ProblemSpec(BASKET_HESTON, _build_basket_heston, calibrate_basket_heston)
)
