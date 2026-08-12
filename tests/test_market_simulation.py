import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from market_simulation.black_scholes import (
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
from conftest import bs_mc_feature_price
from market_simulation.black_scholes_mc import (
    MC_NUM_STEPS,
    simulate_terminal,
    generate_training_paths,
)
from market_simulation.monte_carlo_pricer import MonteCarloPricer
from market_simulation.payoff import (
    AsianCall,
    EuropeanCall,
    EuropeanPut,
    cubic_spline_smooth,
    relu,
    sigmoid_smooth,
)
from market_simulation.pricing_model import (
    BachelierModel,
    BachelierParams,
    BlackScholesModel,
    BlackScholesParams,
)
from market_simulation.timesteppingscheme import EulerMaruyama, Milstein

S0, K, T, SIGMA, R = 100.0, 100.0, 1.0, 0.2, 0.05


def _brownian(key, num_paths, num_steps, dt, dim=1):
    """
    Mirrors TimeSteppingScheme.generate_paths so a run can be compared against
    the exact solution driven by the same increments.
    """
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
        float(black_scholes_price_single(S0, K, T, s, R)) for s in [0.1, 0.2, 0.4, 0.8]
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
            S0,
            float(strikes[i]),
            float(maturities[i]),
            SIGMA,
            R,
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
        s0=s0,
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=jax.random.PRNGKey(0),
        num_paths=32,
        num_steps=10,
        dt=T / 10,
    )

    assert paths.shape == (32, 11, 1)
    assert bool(jnp.all(paths[:, 0, 0] == S0))


def test_milstein_is_more_accurate_than_euler_on_gbm():
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
            s0=s0,
            drift_fn=model.drift,
            diffusion_fn=model.diffusion,
            params=params,
            key=key,
            num_paths=num_paths,
            num_steps=num_steps,
            dt=dt,
        )

        errors[name] = float(jnp.mean(jnp.abs(paths[:, -1, 0] - exact)))

    assert errors["milstein"] < errors["euler"]


def test_discounted_spot_is_a_martingale():
    model = BlackScholesModel(scheme=EulerMaruyama())
    params = BlackScholesParams(r=R, sigma=SIGMA)

    paths = model.scheme.generate_paths(
        s0=jnp.array([S0]),
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=jax.random.PRNGKey(1),
        num_paths=60_000,
        num_steps=50,
        dt=T / 50,
    )

    discounted = float(jnp.mean(paths[:, -1, 0]) * jnp.exp(-R * T))

    assert abs(discounted - S0) < 0.5


def test_correlation_matrix_is_applied_to_the_increments():
    rho = 0.7
    model = BachelierModel(scheme=EulerMaruyama())
    params = BachelierParams(sigma=1.0)

    paths = model.scheme.generate_paths(
        s0=jnp.zeros(2),
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=jax.random.PRNGKey(2),
        num_paths=40_000,
        num_steps=20,
        dt=T / 20,
        corr=jnp.array([[1.0, rho], [rho, 1.0]]),
    )

    terminal = paths[:, -1, :]
    empirical = float(jnp.corrcoef(terminal.T)[0, 1])

    assert abs(empirical - rho) < 0.02


def test_uncorrelated_by_default():
    model = BachelierModel(scheme=EulerMaruyama())

    paths = model.scheme.generate_paths(
        s0=jnp.zeros(2),
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=BachelierParams(sigma=1.0),
        key=jax.random.PRNGKey(3),
        num_paths=40_000,
        num_steps=20,
        dt=T / 20,
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
        s0=jnp.array([S0]),
        params=params,
        maturity=T,
        num_paths=80_000,
        num_steps=50,
        key=jax.random.PRNGKey(0),
    )

    price = float(undiscounted * jnp.exp(-R * T))
    analytic = float(black_scholes_price_single(S0, K, T, SIGMA, R))

    assert abs(price - analytic) < 0.3


def test_value_fn_selects_the_underlying():
    model = BlackScholesModel(scheme=EulerMaruyama())
    params = BlackScholesParams(r=R, sigma=SIGMA)

    doubled = MonteCarloPricer(
        model, EuropeanCall(strike=2 * K, smooth_fn=relu), value_fn=lambda s: 2.0 * s[0]
    ).price(
        s0=jnp.array([S0]),
        params=params,
        maturity=T,
        num_paths=20_000,
        num_steps=25,
        key=jax.random.PRNGKey(0),
    )

    plain = MonteCarloPricer(model, EuropeanCall(strike=K, smooth_fn=relu)).price(
        s0=jnp.array([S0]),
        params=params,
        maturity=T,
        num_paths=20_000,
        num_steps=25,
        key=jax.random.PRNGKey(0),
    )

    assert abs(float(doubled) - 2.0 * float(plain)) < 1e-8


def test_bs_mc_price_matches_analytic_and_is_reproducible():
    x = jnp.array([S0, K, T, SIGMA, R])

    key = jax.random.PRNGKey(0)

    first = float(bs_mc_feature_price(x, key))
    second = float(bs_mc_feature_price(x, key))
    analytic = float(black_scholes_price_single(S0, K, T, SIGMA, R))

    assert first == second, "the same key must give the same label"
    assert abs(first - analytic) < 0.5

    fresh = float(bs_mc_feature_price(x, jax.random.PRNGKey(123)))

    assert fresh != first, "a different key must re-draw the paths"
    assert abs(fresh - analytic) < 0.5


def test_exact_terminal_sampling_is_unbiased():
    x = jnp.array([S0, K, T, SIGMA, R])

    prices = jnp.array(
        [float(bs_mc_feature_price(x, jax.random.PRNGKey(s))) for s in range(24)]
    )

    analytic = float(black_scholes_price_single(S0, K, T, SIGMA, R))

    relative = float(jnp.mean(prices)) / analytic - 1.0

    assert -0.01 < relative <= 0.001, f"unexpected sampling bias {relative:+.4%}"


def test_antithetic_sampling_reduces_variance():
    x = jnp.array([S0, K, T, SIGMA, R])

    def spread(antithetic):
        return float(
            jnp.std(
                jnp.array(
                    [
                        float(
                            bs_mc_feature_price(
                                x,
                                jax.random.PRNGKey(s),
                                num_paths=4_000,
                                antithetic=antithetic,
                            )
                        )
                        for s in range(24)
                    ]
                )
            )
        )

    assert spread(True) < spread(False)


def test_terminal_draws_are_lognormal_with_the_right_moments():
    blocks = simulate_terminal(
        S0, SIGMA, R, T, 200_000, jax.random.PRNGKey(3), antithetic=False
    )

    terminal = blocks[0]

    assert len(blocks) == 1
    assert bool(jnp.all(terminal > 0.0))

    forward = S0 * math.exp(R * T)

    assert abs(float(jnp.mean(terminal)) / forward - 1.0) < 0.01


def test_mc_delta_matches_analytic():
    x = jnp.array([S0, K, T, SIGMA, R])

    mc_delta = float(jax.grad(bs_mc_feature_price)(x, jax.random.PRNGKey(0))[0])

    assert abs(mc_delta - float(delta(S0, K, T, SIGMA, R))) < 0.05


def test_generate_training_paths_grid():
    x = jnp.array([S0, K, T, SIGMA, R])

    time_grid, paths = generate_training_paths(x, num_paths=16)

    assert time_grid.shape == (MC_NUM_STEPS + 1,)
    assert paths.shape == (16, MC_NUM_STEPS + 1, 1)
    assert abs(float(time_grid[-1]) - T) < 1e-12


def test_hvp_dataset_respects_sobolev_order():
    from market_simulation.sobolev_labels import create_sobolev_labels, label_keys

    X = jnp.tile(jnp.array([S0, K, T, SIGMA, R]), (2, 1))
    V = jnp.tile(jnp.eye(5)[0], (2, 1))
    keys = label_keys(0, 2)

    price_fn = lambda x, key: bs_mc_feature_price(x, key, num_paths=2_000)

    prices, gradients, hvps = create_sobolev_labels(
        price_fn, X, V, keys, sobolev_order=2
    )

    assert prices.shape == (2,)
    assert gradients.shape == (2, 5)
    assert hvps.shape == (2, 5)

    _, _, no_hvps = create_sobolev_labels(price_fn, X, V, keys, sobolev_order=1)

    assert no_hvps is None


def test_label_keys_are_independent_unless_asked_otherwise():
    from market_simulation.sobolev_labels import label_keys

    independent = label_keys(0, 8)
    shared = label_keys(0, 8, shared=True)

    assert independent.shape == (8, 2)
    assert not bool(jnp.all(independent == independent[0]))
    assert bool(jnp.all(shared == shared[0]))


def test_labels_reject_a_key_count_mismatch():
    from market_simulation.sobolev_labels import create_sobolev_labels, label_keys

    X = jnp.tile(jnp.array([S0, K, T, SIGMA, R]), (3, 1))
    V = jnp.tile(jnp.eye(5)[0], (3, 1))

    try:
        create_sobolev_labels(
            lambda x, key: bs_mc_feature_price(x, key, num_paths=500),
            X,
            V,
            label_keys(0, 2),
            sobolev_order=1,
        )
    except ValueError as e:
        assert "one key per sample" in str(e)
    else:
        assert False, "a key/sample mismatch must raise"


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
        test_exact_terminal_sampling_is_unbiased,
        test_antithetic_sampling_reduces_variance,
        test_terminal_draws_are_lognormal_with_the_right_moments,
        test_label_keys_are_independent_unless_asked_otherwise,
        test_labels_reject_a_key_count_mismatch,
        test_mc_delta_matches_analytic,
        test_generate_training_paths_grid,
        test_hvp_dataset_respects_sobolev_order,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
