import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
from marktsimulation.black_scholes import black_scholes_price_single
from marktsimulation.heston import (
    feller_ratio,
    heston_characteristic,
    heston_price,
    heston_price_vector,
)
from marktsimulation.pricing_model import (
    HestonModel,
    HestonParams,
    VARIANCE_SMOOTHING,
    smooth_positive,
)
from marktsimulation.timesteppingscheme import EulerMaruyama
from pipeline.config import (
    BasketConfig,
    DataConfig,
    ExperimentConfig,
    HestonConfig,
    PayoffConfig,
    SimulationConfig,
)
from surrogate_modeling.problems.heston import calibrate_heston
from surrogate_modeling.pricing_problem import (
    CalibrationResult,
    available_problems,
    build_problem,
)

S0, K, T, R = 100.0, 100.0, 1.0, 0.03

P = HestonParams(r=R, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, nu0=0.04)

HESTON = "heston"
BASKET_HESTON = "basket_heston"

ALL_MODELS = (
    "black_scholes",
    "basket_black_scholes",
    "bachelier",
    "basket_bachelier",
    "heston",
    "basket_heston",
)


def _market_data(spot=100.0):
    return MarketData(
        spot=spot,
        strikes=jnp.array([80.0, 100.0, 120.0]),
        maturities=jnp.array([0.25, 0.5, 1.0]),
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )


def _params_for(model):
    """Each model family has its own parameter shape."""

    from marktsimulation.pricing_model import BachelierParams, BlackScholesParams

    if "heston" in model:
        return P

    if "bachelier" in model:
        return BachelierParams(sigma=20.0, r=R)

    return BlackScholesParams(r=R, sigma=0.2)


def _problem(model, payoff="european_call", params=None, paths=20_000, steps=32):
    config = ExperimentConfig(
        payoff=PayoffConfig(name=payoff),
        simulation=SimulationConfig(
            num_paths=paths, num_steps=steps, reference_paths=4 * paths
        ),
        data=DataConfig(pricing_model=model, min_maturity=0.05),
        basket=BasketConfig(n_assets=3),
        heston=HestonConfig(),
    )

    return build_problem(
        model,
        config=config,
        market_data=_market_data(),
        calibration=CalibrationResult(params=params or _params_for(model)),
    )


def test_zero_vol_of_vol_reduces_to_black_scholes():
    """
    With xi -> 0 the variance follows a deterministic ODE, so the price is
    Black-Scholes on the integrated variance. This pins the characteristic
    function against an independent formula.
    """

    for kappa, theta, v0, maturity in [
        (2.0, 0.04, 0.04, 1.0),
        (2.0, 0.04, 0.09, 1.0),
        (1.0, 0.09, 0.01, 2.0),
        (3.0, 0.02, 0.05, 0.5),
    ]:
        params = HestonParams(r=R, kappa=kappa, theta=theta, xi=1e-6, rho=0.0, nu0=v0)

        integrated = (
            theta * maturity + (v0 - theta) * (1 - math.exp(-kappa * maturity)) / kappa
        )

        reference = float(
            black_scholes_price_single(
                S0, K, maturity, math.sqrt(integrated / maturity), R, True
            )
        )

        assert (
            abs(float(heston_price(S0, K, maturity, params, True)) - reference) < 1e-4
        )


def test_put_call_parity():
    for strike in (70.0, 100.0, 140.0):
        call = float(heston_price(S0, strike, T, P, True))
        put = float(heston_price(S0, strike, T, P, False))

        assert abs((call - put) - (S0 - strike * math.exp(-R * T))) < 1e-10


def test_the_quadrature_has_converged():
    for maturity in (0.1, 1.0, 3.0):
        prices = [
            float(heston_price(S0, K, maturity, P, True, u_max=limit))
            for limit in (200.0, 400.0, 800.0)
        ]

        assert max(prices) - min(prices) < 1e-6


def test_characteristic_function_at_minus_i_is_the_forward():
    value = heston_characteristic(-1j, S0, T, P)

    assert abs(complex(value).real - S0 * math.exp(R * T)) < 1e-8
    assert abs(complex(value).imag) < 1e-8


def test_price_stays_inside_its_model_free_bounds():
    for strike in (60.0, 100.0, 160.0):
        price = float(heston_price(S0, strike, T, P, True))

        assert max(S0 - strike * math.exp(-R * T), 0.0) - 1e-9 <= price <= S0


def test_the_analytic_price_is_twice_differentiable():
    def price(x):
        return heston_price(
            x[0],
            K,
            T,
            HestonParams(r=R, kappa=x[1], theta=x[2], xi=x[3], rho=x[4], nu0=x[5]),
            True,
        )

    x = jnp.array([S0, 2.0, 0.04, 0.5, -0.7, 0.04])

    gradient = jax.grad(price)(x)
    hvp = jax.jvp(jax.grad(price), (x,), (jnp.ones(6) / math.sqrt(6),))[1]

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(hvp)))

    for i in range(len(x)):
        h = 1e-5 * max(abs(float(x[i])), 1.0)
        fd = (float(price(x.at[i].add(h))) - float(price(x.at[i].add(-h)))) / (2 * h)

        assert abs(float(gradient[i]) - fd) < 1e-5 * max(abs(fd), 1.0)


def test_vectorised_prices_match_the_scalar_form():
    strikes = jnp.array([80.0, 100.0, 120.0, 100.0])
    maturities = jnp.array([0.5, 0.5, 0.5, 1.0])
    is_call = jnp.array([True, True, False, False])

    vector = heston_price_vector(P, strikes, maturities, is_call, S0)

    for i in range(4):
        scalar = float(
            heston_price(
                S0, float(strikes[i]), float(maturities[i]), P, bool(is_call[i])
            )
        )

        assert abs(float(vector[i]) - scalar) < 1e-10


def test_the_smooth_positive_part_is_twice_differentiable():
    """The property `jnp.maximum(v, 0)` lacks: a finite second derivative."""

    width = 1e-3

    f = lambda v: smooth_positive(v, width)

    for v in (0.05, 1e-3, 0.0, -1e-3, -0.05):
        assert float(f(v)) > 0.0
        assert math.isfinite(float(jax.grad(f)(v)))
        assert math.isfinite(float(jax.grad(jax.grad(f))(v)))

    assert abs(float(f(1.0)) - 1.0) < width**2
    assert float(f(-1.0)) < width**2

    assert abs(float(jax.grad(jax.grad(f))(0.0)) - 0.5 / width) < 1e-6


def test_the_positive_part_bounds_the_square_root_derivative():
    """The property that actually fixes the gradient: strict positivity."""

    width = 1e-3

    bound = 1.0 / (2.0 * math.sqrt(width / 2.0))

    smooth_sqrt = lambda v: jnp.sqrt(smooth_positive(v, width))
    hard_sqrt = lambda v: jnp.sqrt(jnp.maximum(v, 0.0) + 1e-300)
    reflect_sqrt = lambda v: jnp.sqrt(jnp.abs(v) + 1e-300)

    for v in (0.04, 1e-3, 1e-6, 0.0, -1e-6, -0.04):
        assert abs(float(jax.grad(smooth_sqrt)(v))) <= bound + 1e-9, v

    assert abs(float(jax.grad(hard_sqrt)(1e-6))) > 10.0 * bound
    assert abs(float(jax.grad(reflect_sqrt)(1e-6))) > 10.0 * bound


def test_the_variance_entering_the_square_root_is_always_positive():
    model = HestonModel(scheme=EulerMaruyama())

    for kappa, xi in [(2.0, 0.2), (2.0, 0.5), (1.0, 0.7)]:
        params = HestonParams(r=R, kappa=kappa, theta=0.04, xi=xi, rho=-0.7, nu0=0.04)

        paths = model.scheme.generate_paths(
            s0=jnp.array([S0, params.nu0]),
            drift_fn=model.drift,
            diffusion_fn=model.diffusion,
            params=params,
            key=jax.random.PRNGKey(1),
            num_paths=4_000,
            num_steps=50,
            dt=1.0 / 50,
            corr=model.noise_correlation(params),
        )

        used = smooth_positive(paths[:, :, 1], VARIANCE_SMOOTHING * params.theta)

        assert bool(jnp.all(used > 0.0)), (kappa, xi)
        assert bool(jnp.all(jnp.isfinite(paths)))

        assert bool(jnp.all(paths[:, :, 0] > 0.0)), (kappa, xi)


def test_the_raw_variance_state_does_go_negative_when_feller_is_violated():
    model = HestonModel(scheme=EulerMaruyama())

    params = HestonParams(r=R, kappa=1.0, theta=0.04, xi=0.7, rho=-0.7, nu0=0.04)

    assert feller_ratio(params) < 1.0

    paths = model.scheme.generate_paths(
        s0=jnp.array([S0, params.nu0]),
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=jax.random.PRNGKey(1),
        num_paths=4_000,
        num_steps=50,
        dt=1.0 / 50,
        corr=model.noise_correlation(params),
    )

    assert float(jnp.min(paths[:, :, 1])) < 0.0


def test_feller_ratio_is_reported_correctly():
    assert abs(feller_ratio(P) - 2 * 2.0 * 0.04 / 0.25) < 1e-12

    assert (
        feller_ratio(
            HestonParams(r=R, kappa=2.0, theta=0.04, xi=0.2, rho=0.0, nu0=0.04)
        )
        > 1.0
    )


def test_hard_truncation_destroys_the_variance_sensitivity():
    """Regression test for the scheme choice."""

    hard = lambda v: jnp.maximum(v, 0.0)
    soft = lambda v: smooth_positive(v, 1e-3)

    assert abs(float(hard(0.05)) - float(soft(0.05))) < 1e-3**2 / (4 * 0.05) * 1.01

    for v in (-0.01, 0.0, 0.01):
        assert float(jax.grad(jax.grad(hard))(v)) == 0.0

    assert float(jax.grad(jax.grad(soft))(0.0)) > 0.0

    assert float(jax.grad(hard)(-0.01)) == 0.0
    assert float(jax.grad(soft)(-0.01)) > 0.0


def test_monte_carlo_labels_agree_with_the_fourier_reference():
    problem = _problem(HESTON, paths=40_000, steps=64)

    x = problem.baseline_features()

    analytic = float(problem.analytic_price(x))

    draws = jnp.array(
        [float(problem.label_price_fn()(x, jax.random.PRNGKey(s))) for s in range(10)]
    )

    standard_error = float(jnp.std(draws)) / math.sqrt(len(draws))

    assert abs(float(jnp.mean(draws)) - analytic) < 4.0 * standard_error + 0.02


def test_label_gradient_and_hvp_match_finite_differences():
    problem = _problem(HESTON, paths=20_000, steps=32)

    x = problem.baseline_features()
    key = jax.random.PRNGKey(4)

    keyed = lambda z: problem.label_price_fn()(z, key)

    gradient = jax.grad(keyed)(x)

    v = jax.random.normal(jax.random.PRNGKey(9), (8,))
    v = v / jnp.linalg.norm(v)

    hvp = jax.jvp(jax.grad(keyed), (x,), (v,))[1]
    hvp_fd = (jax.grad(keyed)(x + 1e-4 * v) - jax.grad(keyed)(x - 1e-4 * v)) / 2e-4

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(hvp)))

    for i in range(8):
        h = 1e-4 * max(abs(float(x[i])), 1.0)
        fd = (float(keyed(x.at[i].add(h))) - float(keyed(x.at[i].add(-h)))) / (2 * h)

        assert abs(float(gradient[i]) - fd) < 1e-3 * max(abs(fd), 1.0), i
        assert abs(float(hvp[i]) - float(hvp_fd[i])) < 1e-3 * max(
            abs(float(hvp_fd[i])), 1.0
        ), i


def test_the_monte_carlo_gradient_tracks_the_analytic_one():
    problem = _problem(HESTON, paths=40_000, steps=64)

    x = problem.baseline_features()

    analytic = jax.grad(lambda z: problem.analytic_price(z))(x)

    simulated = jnp.mean(
        jnp.stack(
            [
                jax.grad(lambda z: problem.label_price_fn()(z, jax.random.PRNGKey(s)))(
                    x
                )
                for s in range(8)
            ]
        ),
        axis=0,
    )

    for index in (0, 1, 2, 3):
        assert abs(float(simulated[index]) - float(analytic[index])) < 0.05 * max(
            abs(float(analytic[index])), 1.0
        ), index


def test_basket_heston_symmetry_and_shape():
    problem = _problem(BASKET_HESTON, paths=10_000, steps=24)

    assert problem.feature_names == ("S1", "S2", "S3", "K", "T")
    assert problem.exchangeable_features == (0, 1, 2)

    key = jax.random.PRNGKey(2)

    values = [
        float(problem.label_price_fn()(jnp.array(list(s) + [100.0, 0.5]), key))
        for s in [(90.0, 100.0, 115.0), (115.0, 90.0, 100.0), (100.0, 115.0, 90.0)]
    ]

    assert max(values) - min(values) == 0.0

    x = problem.baseline_features()

    gradient = jax.grad(lambda z: problem.label_price_fn()(z, key))(x)

    assert bool(jnp.all((gradient[:3] >= 0.0) & (gradient[:3] <= 1.0 / 3.0 + 1e-9)))
    assert float(gradient[3]) < 0.0


def test_basket_heston_declares_no_closed_form():
    problem = _problem(BASKET_HESTON, paths=4_000, steps=16)

    assert problem.analytic_price(problem.baseline_features()) is None


def test_the_basket_correlation_block_is_positive_semidefinite():
    problem = _problem(BASKET_HESTON, paths=4_000, steps=16)

    corr = problem.model.noise_correlation(problem.params)

    assert corr.shape == (6, 6)
    assert float(jnp.min(jnp.linalg.eigvalsh(corr))) > -1e-12


def test_calibration_recovers_synthetic_heston_prices():
    truth = HestonParams(r=0.03, kappa=1.5, theta=0.05, xi=0.4, rho=-0.6, nu0=0.045)

    strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0] * 3)
    maturities = jnp.concatenate(
        [jnp.full(5, 0.25), jnp.full(5, 0.75), jnp.full(5, 1.5)]
    )
    is_call = jnp.array([True] * 15)

    spot = 100.0

    synthetic = MarketData(
        spot=spot,
        strikes=strikes,
        maturities=maturities,
        market_prices=heston_price_vector(truth, strikes, maturities, is_call, spot),
        is_call=is_call,
    )

    config = ExperimentConfig(
        data=DataConfig(pricing_model=HESTON),
        heston=HestonConfig(initial_kappa=2.0, initial_xi=0.5, initial_rho=-0.7),
    )

    result = calibrate_heston(config, synthetic)

    fitted_prices = heston_price_vector(
        result.params, strikes, maturities, is_call, spot
    )

    assert float(jnp.max(jnp.abs(fitted_prices - synthetic.market_prices))) < 0.05

    assert float(result.params.kappa) > 0.0
    assert float(result.params.theta) > 0.0
    assert float(result.params.xi) > 0.0
    assert abs(float(result.params.rho)) < 1.0
    assert float(result.params.nu0) > 0.0

    assert "feller_ratio" in result.diagnostics
    assert result.assumptions == {}

    for key in (
        "residual_rmse",
        "residual_median_relative_pct",
        "residual_mean_signed",
        "residual_max_absolute",
    ):
        assert key in result.diagnostics, key
        assert math.isfinite(float(result.diagnostics[key])), key


def test_the_basket_variant_declares_every_structural_assumption():
    from surrogate_modeling.problems.heston import calibrate_basket_heston

    config = ExperimentConfig(
        data=DataConfig(pricing_model=BASKET_HESTON),
        basket=BasketConfig(n_assets=3),
        heston=HestonConfig(),
    )

    result = calibrate_basket_heston(config, _market_data())

    assert "asset_correlation" in result.assumptions
    assert "spot_variance_cross_correlation" in result.assumptions
    assert "per_asset_dynamics" in result.assumptions


def test_mc_price_refuses_to_guess_the_underlying():
    """
    A multi-dimensional state with no `value_fn` used to fall through to
    `block[:, 0]`, which is right for Heston only by accident of ordering.
    """

    from marktsimulation.mc_pricing import mc_price

    model = HestonModel(scheme=EulerMaruyama())

    try:
        mc_price(
            model,
            P,
            jnp.array([S0, P.nu0]),
            K,
            T,
            jax.random.PRNGKey(0),
            num_paths=64,
            num_steps=4,
        )
    except ValueError as e:
        assert "value_fn" in str(e)
    else:
        assert False, "an ambiguous multi-dimensional state must raise"

    price = mc_price(
        model,
        P,
        jnp.array([S0, P.nu0]),
        K,
        T,
        jax.random.PRNGKey(0),
        num_paths=64,
        num_steps=4,
        value_fn=lambda state: state[0],
    )

    assert math.isfinite(float(price))


def test_the_unconstrained_reparametrisation_round_trips():
    from surrogate_modeling.problems.heston import _heston_transforms

    to_model, to_unconstrained = _heston_transforms()

    recovered = to_model(to_unconstrained(P))

    for name in ("r", "kappa", "theta", "xi", "rho", "nu0"):
        assert abs(float(getattr(recovered, name)) - float(getattr(P, name))) < 1e-10


def test_all_six_models_are_registered():
    assert set(available_problems()) == set(ALL_MODELS)


def test_every_model_exposes_a_consistent_interface():
    for model in ALL_MODELS:
        problem = _problem(model, paths=2_000, steps=8)

        low, high = problem.feature_bounds()
        baseline = problem.baseline_features()

        assert len(problem.feature_names) == problem.n_features, model
        assert len(problem.feature_labels) == problem.n_features, model
        assert baseline.shape == (problem.n_features,), model
        assert bool(jnp.all(baseline >= low - 1e-9)), model
        assert bool(jnp.all(baseline <= high + 1e-9)), model
        assert len(problem.surface_specs()) == problem.n_features - 1, model

        points = problem.reference_points(n_points=8)

        assert bool(jnp.all(points >= low - 1e-9)), model
        assert bool(jnp.all(points <= high + 1e-9)), model


def test_the_heston_problem_carries_its_parameters_as_features():
    problem = _problem(HESTON, paths=2_000, steps=8)

    assert problem.feature_names == ("S", "K", "T", "v0", "kappa", "theta", "xi", "rho")
    assert problem.n_features == 8


def test_heston_exposure_paths_respect_the_training_floor():
    for model in (HESTON, BASKET_HESTON):
        problem = _problem(model, paths=2_000, steps=8)

        _, features = problem.exposure_paths(
            strike=100.0, horizon=1.0, num_paths=4, num_steps=8, min_maturity=0.05
        )

        assert features.shape == (4, 9, problem.n_features), model

        maturity = features[:, :, problem.feature_names.index("T")]

        assert float(jnp.min(maturity)) >= 0.05 - 1e-12, model


def test_payoff_switching_on_heston():
    european = _problem(HESTON, payoff="european_call", paths=2_000, steps=8)
    asian = _problem(HESTON, payoff="asian_call", paths=2_000, steps=8)

    assert european.arbitrage_bounds(european.baseline_features()) is not None
    assert asian.arbitrage_bounds(asian.baseline_features()) is None
    assert asian.shape_constraints() == ()

    put = _problem(HESTON, payoff="european_put", paths=2_000, steps=8)

    x = put.baseline_features()

    call_price = float(european.analytic_price(x))
    put_price = float(put.analytic_price(x))

    assert (
        abs(
            (call_price - put_price)
            - (float(x[0]) - float(x[1]) * math.exp(-P.r * float(x[2])))
        )
        < 1e-8
    )


if __name__ == "__main__":
    for check in [
        test_zero_vol_of_vol_reduces_to_black_scholes,
        test_put_call_parity,
        test_the_quadrature_has_converged,
        test_characteristic_function_at_minus_i_is_the_forward,
        test_price_stays_inside_its_model_free_bounds,
        test_the_analytic_price_is_twice_differentiable,
        test_vectorised_prices_match_the_scalar_form,
        test_the_smooth_positive_part_is_twice_differentiable,
        test_the_variance_entering_the_square_root_is_always_positive,
        test_the_raw_variance_state_does_go_negative_when_feller_is_violated,
        test_feller_ratio_is_reported_correctly,
        test_hard_truncation_destroys_the_variance_sensitivity,
        test_monte_carlo_labels_agree_with_the_fourier_reference,
        test_label_gradient_and_hvp_match_finite_differences,
        test_the_monte_carlo_gradient_tracks_the_analytic_one,
        test_basket_heston_symmetry_and_shape,
        test_basket_heston_declares_no_closed_form,
        test_the_basket_correlation_block_is_positive_semidefinite,
        test_calibration_recovers_synthetic_heston_prices,
        test_the_basket_variant_declares_every_structural_assumption,
        test_mc_price_refuses_to_guess_the_underlying,
        test_the_positive_part_bounds_the_square_root_derivative,
        test_the_unconstrained_reparametrisation_round_trips,
        test_all_six_models_are_registered,
        test_every_model_exposes_a_consistent_interface,
        test_the_heston_problem_carries_its_parameters_as_features,
        test_heston_exposure_paths_respect_the_training_floor,
        test_payoff_switching_on_heston,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
