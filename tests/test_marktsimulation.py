import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from marktsimulation.black_scholes import (
    black_scholes_price,
    black_scholes_price_single,
    bs_feature_gradient,
    bs_feature_hessian,
    bs_feature_price,
    compare_to_mc,
    create_bs_dataset,
    delta,
    gamma,
    vega,
)
from marktsimulation.black_scholes_mc import (
    MC_NUM_STEPS,
    bs_mc_price,
    create_bs_mc_dataset_with_hvps,
    generate_training_paths,
)
from marktsimulation.monte_carlo_pricer import MonteCarloPricer
from marktsimulation.payoff import (
    AsianCall,
    EuropeanCall,
    EuropeanPut,
    cubic_spline_smooth,
    relu,
    sigmoid_smooth,
)
from marktsimulation.pricing_model import (
    BachelierModel,
    BachelierParams,
    BlackScholesModel,
    BlackScholesParams,
)
from marktsimulation.timesteppingscheme import EulerMaruyama, Milstein


S0, K, T, SIGMA, R = 100.0, 100.0, 1.0, 0.2, 0.05


def _brownian(key, num_paths, num_steps, dt, dim=1):
    """Mirrors TimeSteppingScheme.generate_paths so a run can be compared
    against the exact solution driven by the same increments."""
    return jax.random.normal(key, (num_paths, num_steps, dim)) * jnp.sqrt(dt)


def test_put_call_parity():
    call = float(black_scholes_price_single(S0, K, T, SIGMA, R, is_call=True))
    put = float(black_scholes_price_single(S0, K, T, SIGMA, R, is_call=False))

    assert abs((call - put) - (S0 - K * jnp.exp(-R * T))) < 1e-10


def test_price_above_intrinsic_and_below_spot():
    call = float(black_scholes_price_single(S0, 80.0, T, SIGMA, R))

    assert call >= S0 - 80.0 * float(jnp.exp(-R * T)) - 1e-10
    assert call <= S0


def test_price_increases_with_volatility():
    prices = [
        float(black_scholes_price_single(S0, K, T, s, R))
        for s in [0.1, 0.2, 0.4, 0.8]
    ]

    assert all(a < b for a, b in zip(prices, prices[1:]))


def test_deep_itm_call_approaches_discounted_forward():
    call = float(black_scholes_price_single(S0, 1.0, T, SIGMA, R))

    assert abs(call - (S0 - 1.0 * float(jnp.exp(-R * T)))) < 1e-6


def test_vectorized_pricing_matches_single():
    strikes = jnp.array([80.0, 100.0, 120.0])
    maturities = jnp.array([0.5, 1.0, 2.0])
    is_call = jnp.array([True, False, True])

    vector = black_scholes_price(
        BlackScholesParams(r=R, sigma=SIGMA), strikes, maturities, is_call, S0
    )

    for i in range(3):
        single = black_scholes_price_single(
            S0, float(strikes[i]), float(maturities[i]), SIGMA, R,
            is_call=bool(is_call[i]),
        )
        assert abs(float(vector[i]) - float(single)) < 1e-12


def test_greeks_have_expected_signs_and_bounds():
    call_delta = float(delta(S0, K, T, SIGMA, R, is_call=True))
    put_delta = float(delta(S0, K, T, SIGMA, R, is_call=False))

    assert 0.0 < call_delta < 1.0
    assert -1.0 < put_delta < 0.0
    assert abs((call_delta - put_delta) - 1.0) < 1e-12

    assert float(gamma(S0, K, T, SIGMA, R)) > 0.0
    assert float(vega(S0, K, T, SIGMA, R)) > 0.0


def test_autodiff_greeks_match_closed_form():
    x = jnp.array([S0, K, T, SIGMA, R])

    grad = bs_feature_gradient(x)

    assert abs(float(grad[0]) - float(delta(S0, K, T, SIGMA, R))) < 1e-9
    assert abs(float(grad[3]) - float(vega(S0, K, T, SIGMA, R))) < 1e-9

    hessian = bs_feature_hessian(x)

    assert abs(float(hessian[0, 0]) - float(gamma(S0, K, T, SIGMA, R))) < 1e-9
    assert bool(jnp.allclose(hessian, hessian.T))


def test_create_bs_dataset_shapes():
    X = jnp.tile(jnp.array([S0, K, T, SIGMA, R]), (4, 1))

    prices, gradients, hessians = create_bs_dataset(X)

    assert prices.shape == (4,)
    assert gradients.shape == (4, 5)
    assert hessians.shape == (4, 5, 5)
    assert abs(float(prices[0]) - float(bs_feature_price(X[0]))) < 1e-12


def test_compare_to_mc_reports_relative_error():
    result = compare_to_mc(10.0, 11.0)

    assert abs(result["abs_error"] - 1.0) < 1e-12
    assert abs(result["rel_error"] - 0.1) < 1e-12


def test_schemes_keep_shape_and_initial_state():
    model = BlackScholesModel(scheme=EulerMaruyama())
    params = BlackScholesParams(r=R, sigma=SIGMA)
    s0 = jnp.array([S0])

    paths = model.scheme.generate_paths(
        s0=s0, drift_fn=model.drift, diffusion_fn=model.diffusion, params=params,
        key=jax.random.PRNGKey(0), num_paths=32, num_steps=10, dt=T / 10,
    )

    assert paths.shape == (32, 11, 1)
    assert bool(jnp.all(paths[:, 0, 0] == S0))


def test_milstein_is_more_accurate_than_euler_on_gbm():
    # GBM is scored against its closed-form terminal value, driven by the
    # very same Brownian increments
    params = BlackScholesParams(r=R, sigma=SIGMA)
    s0 = jnp.array([S0])
    key = jax.random.PRNGKey(0)

    num_paths, num_steps = 4000, 8
    dt = T / num_steps

    W_T = jnp.sum(_brownian(key, num_paths, num_steps, dt)[:, :, 0], axis=1)
    exact = S0 * jnp.exp((R - 0.5 * SIGMA**2) * T + SIGMA * W_T)

    errors = {}
    for name, scheme in [("euler", EulerMaruyama()), ("milstein", Milstein())]:
        model = BlackScholesModel(scheme=scheme)

        paths = scheme.generate_paths(
            s0=s0, drift_fn=model.drift, diffusion_fn=model.diffusion,
            params=params, key=key, num_paths=num_paths, num_steps=num_steps, dt=dt,
        )

        errors[name] = float(jnp.mean(jnp.abs(paths[:, -1, 0] - exact)))

    assert errors["milstein"] < errors["euler"]


def test_discounted_spot_is_a_martingale():
    model = BlackScholesModel(scheme=EulerMaruyama())
    params = BlackScholesParams(r=R, sigma=SIGMA)

    paths = model.scheme.generate_paths(
        s0=jnp.array([S0]), drift_fn=model.drift, diffusion_fn=model.diffusion,
        params=params, key=jax.random.PRNGKey(1), num_paths=60_000,
        num_steps=50, dt=T / 50,
    )

    discounted = float(jnp.mean(paths[:, -1, 0]) * jnp.exp(-R * T))

    assert abs(discounted - S0) < 0.5


def test_correlation_matrix_is_applied_to_the_increments():
    # Bachelier has constant diffusion, so the terminal state is affine in
    # the Brownian path and the correlation carries through unchanged
    rho = 0.7
    model = BachelierModel(scheme=EulerMaruyama())
    params = BachelierParams(sigma=1.0)

    paths = model.scheme.generate_paths(
        s0=jnp.zeros(2), drift_fn=model.drift, diffusion_fn=model.diffusion,
        params=params, key=jax.random.PRNGKey(2), num_paths=40_000,
        num_steps=20, dt=T / 20,
        corr=jnp.array([[1.0, rho], [rho, 1.0]]),
    )

    terminal = paths[:, -1, :]
    empirical = float(jnp.corrcoef(terminal.T)[0, 1])

    assert abs(empirical - rho) < 0.02


def test_uncorrelated_by_default():
    model = BachelierModel(scheme=EulerMaruyama())

    paths = model.scheme.generate_paths(
        s0=jnp.zeros(2), drift_fn=model.drift, diffusion_fn=model.diffusion,
        params=BachelierParams(sigma=1.0), key=jax.random.PRNGKey(3),
        num_paths=40_000, num_steps=20, dt=T / 20,
    )

    assert abs(float(jnp.corrcoef(paths[:, -1, :].T)[0, 1])) < 0.02


def test_relu_ignores_the_smoothing_width():
    x = jnp.array([-2.0, 0.0, 3.0])

    assert bool(jnp.all(relu(x, 0.0) == jnp.array([0.0, 0.0, 3.0])))
    assert bool(jnp.all(relu(x, 5.0) == relu(x, 0.0)))


def test_smoothed_payoffs_converge_to_relu():
    x = jnp.array([-2.0, -0.5, 0.5, 2.0])

    for smooth_fn in [sigmoid_smooth, cubic_spline_smooth]:
        approx = smooth_fn(x, 1e-4)

        assert float(jnp.max(jnp.abs(approx - relu(x)))) < 1e-3, smooth_fn.__name__


def test_cubic_spline_is_exact_outside_the_smoothing_window():
    w = 0.5

    assert float(cubic_spline_smooth(jnp.array(-1.0), w)) == 0.0
    assert abs(float(cubic_spline_smooth(jnp.array(3.0), w)) - 3.0) < 1e-12


def test_smoothed_payoff_is_twice_differentiable():
    # relu is why the MC labels need smoothing at all: its second derivative
    # is zero everywhere, so HVP labels would collapse
    second = jax.grad(jax.grad(lambda z: sigmoid_smooth(z, 0.5)))(0.3)

    assert float(jnp.abs(second)) > 0.0

    relu_second = jax.grad(jax.grad(lambda z: relu(z)))(0.3)

    assert float(relu_second) == 0.0


def test_call_and_put_have_opposite_intrinsic():
    call = EuropeanCall(strike=K, smooth_fn=relu)
    put = EuropeanPut(strike=K, smooth_fn=relu)

    assert float(call(jnp.array(120.0))) == 20.0
    assert float(put(jnp.array(120.0))) == 0.0
    assert float(call(jnp.array(80.0))) == 0.0
    assert float(put(jnp.array(80.0))) == 20.0


def test_asian_payoff_averages_the_path_excluding_the_initial_state():
    payoff = AsianCall(strike=100.0, smooth_fn=relu)

    path = jnp.array([1000.0, 110.0, 110.0, 110.0])

    assert abs(float(payoff(path)) - 10.0) < 1e-12


def test_monte_carlo_pricer_matches_analytic_call():
    model = BlackScholesModel(scheme=EulerMaruyama())
    params = BlackScholesParams(r=R, sigma=SIGMA)

    pricer = MonteCarloPricer(model, EuropeanCall(strike=K, smooth_fn=relu))

    undiscounted = pricer.price(
        s0=jnp.array([S0]), params=params, maturity=T,
        num_paths=80_000, num_steps=50, key=jax.random.PRNGKey(0),
    )

    price = float(undiscounted * jnp.exp(-R * T))
    analytic = float(black_scholes_price_single(S0, K, T, SIGMA, R))

    assert abs(price - analytic) < 0.3


def test_value_fn_selects_the_underlying():
    model = BlackScholesModel(scheme=EulerMaruyama())
    params = BlackScholesParams(r=R, sigma=SIGMA)

    # a value_fn of 2x must double the effective spot
    doubled = MonteCarloPricer(
        model, EuropeanCall(strike=2 * K, smooth_fn=relu),
        value_fn=lambda s: 2.0 * s[0],
    ).price(
        s0=jnp.array([S0]), params=params, maturity=T,
        num_paths=20_000, num_steps=25, key=jax.random.PRNGKey(0),
    )

    plain = MonteCarloPricer(
        model, EuropeanCall(strike=K, smooth_fn=relu)
    ).price(
        s0=jnp.array([S0]), params=params, maturity=T,
        num_paths=20_000, num_steps=25, key=jax.random.PRNGKey(0),
    )

    assert abs(float(doubled) - 2.0 * float(plain)) < 1e-8


def test_bs_mc_price_matches_analytic_and_is_reproducible():
    x = jnp.array([S0, K, T, SIGMA, R])

    first = float(bs_mc_price(x))
    second = float(bs_mc_price(x))
    analytic = float(black_scholes_price_single(S0, K, T, SIGMA, R))

    assert first == second, "the default key must make labels reproducible"
    assert abs(first - analytic) < 0.5

    fresh = float(bs_mc_price(x, key=jax.random.PRNGKey(123)))

    assert fresh != first
    assert abs(fresh - analytic) < 0.5


def test_mc_delta_matches_analytic():
    x = jnp.array([S0, K, T, SIGMA, R])

    mc_delta = float(jax.grad(bs_mc_price)(x)[0])

    assert abs(mc_delta - float(delta(S0, K, T, SIGMA, R))) < 0.05


def test_generate_training_paths_grid():
    x = jnp.array([S0, K, T, SIGMA, R])

    time_grid, paths = generate_training_paths(x, num_paths=16)

    assert time_grid.shape == (MC_NUM_STEPS + 1,)
    assert paths.shape == (16, MC_NUM_STEPS + 1, 1)
    assert abs(float(time_grid[-1]) - T) < 1e-12


def test_hvp_dataset_respects_sobolev_order():
    X = jnp.tile(jnp.array([S0, K, T, SIGMA, R]), (2, 1))
    V = jnp.tile(jnp.eye(5)[0], (2, 1))

    prices, gradients, hvps = create_bs_mc_dataset_with_hvps(X, V, sobolev_order=2)

    assert prices.shape == (2,)
    assert gradients.shape == (2, 5)
    assert hvps.shape == (2, 5)

    _, _, no_hvps = create_bs_mc_dataset_with_hvps(X, V, sobolev_order=1)

    assert no_hvps is None


if __name__ == "__main__":
    for check in [
        test_put_call_parity,
        test_price_above_intrinsic_and_below_spot,
        test_price_increases_with_volatility,
        test_deep_itm_call_approaches_discounted_forward,
        test_vectorized_pricing_matches_single,
        test_greeks_have_expected_signs_and_bounds,
        test_autodiff_greeks_match_closed_form,
        test_create_bs_dataset_shapes,
        test_compare_to_mc_reports_relative_error,
        test_schemes_keep_shape_and_initial_state,
        test_milstein_is_more_accurate_than_euler_on_gbm,
        test_discounted_spot_is_a_martingale,
        test_correlation_matrix_is_applied_to_the_increments,
        test_uncorrelated_by_default,
        test_relu_ignores_the_smoothing_width,
        test_smoothed_payoffs_converge_to_relu,
        test_cubic_spline_is_exact_outside_the_smoothing_window,
        test_smoothed_payoff_is_twice_differentiable,
        test_call_and_put_have_opposite_intrinsic,
        test_asian_payoff_averages_the_path_excluding_the_initial_state,
        test_monte_carlo_pricer_matches_analytic_call,
        test_value_fn_selects_the_underlying,
        test_bs_mc_price_matches_analytic_and_is_reproducible,
        test_mc_delta_matches_analytic,
        test_generate_training_paths_grid,
        test_hvp_dataset_respects_sobolev_order,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
