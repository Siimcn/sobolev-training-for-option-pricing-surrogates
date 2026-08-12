import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
from marktsimulation.bachelier import (
    bachelier_delta,
    bachelier_forward,
    bachelier_gamma,
    bachelier_price,
    bachelier_price_single,
    bachelier_spot_price,
    bachelier_vega,
    basket_bachelier_greeks,
    basket_bachelier_price,
    basket_bachelier_spot_price,
    basket_normal_volatility,
)
from marktsimulation.basket_mc import uniform_correlation
from marktsimulation.mc_pricing import make_feature_price, mc_price
from marktsimulation.monte_carlo_pricer import MonteCarloPricer
from marktsimulation.payoff import EuropeanCall, sigmoid_smooth
from marktsimulation.pricing_model import (
    BachelierModel,
    BachelierParams,
    BasketBachelierModel,
    BasketBachelierParams,
)
from marktsimulation.timesteppingscheme import EulerMaruyama
from pipeline.config import (
    BasketConfig,
    DataConfig,
    ExperimentConfig,
    PayoffConfig,
    SimulationConfig,
)
from surrogate_modeling.problems import calibrate_bachelier
from surrogate_modeling.pricing_problem import (
    CalibrationResult,
    available_problems,
    build_problem,
)


F, K, T, SIG, R = 100.0, 105.0, 1.0, 20.0, 0.04
N = 3
W = jnp.full(N, 1.0 / N)
SIGMAS = jnp.array([18.0, 20.0, 25.0])
CORR = uniform_correlation(N, 0.5)

BACHELIER = "bachelier"
BASKET_BACHELIER = "basket_bachelier"


def _market_data(spot=100.0):
    return MarketData(
        spot=spot,
        strikes=jnp.array([80.0, 100.0, 120.0]),
        maturities=jnp.array([0.25, 0.5, 1.0]),
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )


def _problem(model, payoff="european_call", sigma=SIG, n_assets=3):
    config = ExperimentConfig(
        payoff=PayoffConfig(name=payoff),
        simulation=SimulationConfig(num_paths=20_000, num_steps=16,
                                    reference_paths=60_000),
        data=DataConfig(pricing_model=model, min_maturity=0.05),
        basket=BasketConfig(n_assets=n_assets),
    )

    return build_problem(
        model,
        config=config,
        market_data=_market_data(),
        calibration=CalibrationResult(params=BachelierParams(sigma=sigma, r=R)),
    )


def test_put_call_parity():
    call = float(bachelier_price_single(F, K, T, SIG, R, True))
    put = float(bachelier_price_single(F, K, T, SIG, R, False))

    assert abs((call - put) - float(jnp.exp(-R * T) * (F - K))) < 1e-12


def test_spot_quoting_carries_the_forward():
    spot = 100.0

    assert abs(
        float(bachelier_spot_price(spot, K, T, SIG, R, True))
        - float(bachelier_price_single(spot * math.exp(R * T), K, T, SIG, R, True))
    ) < 1e-12

    deep = float(bachelier_spot_price(spot, 1.0, T, SIG, R, True))

    time_value = deep - (spot - 1.0 * math.exp(-R * T))

    assert 0.0 < time_value < 1e-5


def test_price_is_increasing_in_volatility_and_bounded_below():
    low = float(bachelier_price_single(F, K, T, 10.0, R, True))
    high = float(bachelier_price_single(F, K, T, 40.0, R, True))

    assert high > low
    assert low >= max(math.exp(-R * T) * (F - K), 0.0) - 1e-12


def test_a_normal_underlying_admits_negative_states():
    model = BachelierModel(scheme=EulerMaruyama())

    blocks = model.terminal_state(
        jnp.array([1.0]), BachelierParams(sigma=50.0, r=0.0), 1.0,
        20_000, jax.random.PRNGKey(0), antithetic=False,
    )

    assert bool(jnp.any(blocks[0] < 0.0))


def test_closed_form_greeks_match_autodiff():
    f = lambda x: bachelier_price_single(x[0], x[1], x[2], x[3], x[4], True)

    x = jnp.array([F, K, T, SIG, R])

    g = jax.grad(f)(x)
    h = jax.hessian(f)(x)

    assert abs(float(g[0]) - float(bachelier_delta(F, K, T, SIG, R, True))) < 1e-12
    assert abs(float(h[0, 0]) - float(bachelier_gamma(F, K, T, SIG, R))) < 1e-12
    assert abs(float(g[3]) - float(bachelier_vega(F, K, T, SIG, R))) < 1e-12

    assert abs(float(g[4]) + T * float(f(x))) < 1e-12


def test_basket_greeks_match_autodiff_and_the_hessian_is_rank_one():
    spots = jnp.array([95.0, 100.0, 110.0])

    closed = basket_bachelier_greeks(spots, K, T, W, SIGMAS, CORR, R, True)

    f = lambda s: basket_bachelier_price(s, K, T, W, SIGMAS, CORR, R, True)

    assert float(jnp.max(jnp.abs(jax.grad(f)(spots) - closed["delta"]))) < 1e-12

    hessian = jax.hessian(f)(spots)

    assert float(jnp.max(jnp.abs(hessian - closed["gamma"]))) < 1e-12

    eigenvalues = jnp.sort(jnp.abs(jnp.linalg.eigvalsh(hessian)))

    assert float(eigenvalues[-2]) < 1e-12 * float(eigenvalues[-1] + 1e-30)


def test_basket_price_depends_on_the_spots_only_through_the_basket():
    a = jnp.array([90.0, 100.0, 110.0])
    b = jnp.array([100.0, 100.0, 100.0])

    assert abs(float(jnp.sum(W * a)) - float(jnp.sum(W * b))) < 1e-12

    assert abs(
        float(basket_bachelier_price(a, K, T, W, SIGMAS, CORR, R, True))
        - float(basket_bachelier_price(b, K, T, W, SIGMAS, CORR, R, True))
    ) < 1e-12


def test_feature_pricer_gradient_and_hvp_match_finite_differences():
    model = BasketBachelierModel(scheme=EulerMaruyama())
    params = BasketBachelierParams(r=R, sigmas=SIGMAS, weights=W, corr=CORR)

    price_fn = make_feature_price(model, params, N, num_paths=40_000,
                                  smooth_fraction=0.05, symmetrize=False)

    key = jax.random.PRNGKey(7)
    keyed = lambda z: price_fn(z, key)

    x = jnp.array([95.0, 100.0, 110.0, K, T])

    gradient = jax.grad(keyed)(x)

    v = jnp.array([0.3, -0.5, 0.2, 0.6, -0.4])
    v = v / jnp.linalg.norm(v)

    hvp = jax.jvp(jax.grad(keyed), (x,), (v,))[1]
    hvp_fd = (jax.grad(keyed)(x + 1e-3 * v) - jax.grad(keyed)(x - 1e-3 * v)) / 2e-3

    for i in range(5):
        h = 1e-3 * max(abs(float(x[i])), 1.0)
        fd = (float(keyed(x.at[i].add(h))) - float(keyed(x.at[i].add(-h)))) / (2 * h)

        assert abs(float(gradient[i]) - fd) < 1e-4 * max(abs(fd), 1.0)
        assert abs(float(hvp[i]) - float(hvp_fd[i])) < 1e-4 * max(
            abs(float(hvp_fd[i])), 1.0
        )

    assert bool(jnp.all((gradient[:3] >= 0.0) & (gradient[:3] <= 1.0 / 3.0 + 1e-9)))
    assert float(gradient[3]) < 0.0


def test_exact_sampling_is_unbiased_against_the_closed_form():
    model = BachelierModel(scheme=EulerMaruyama())
    params = BachelierParams(sigma=SIG, r=R)

    prices = jnp.array([
        float(mc_price(model, params, jnp.array([F]), K, T,
                       jax.random.PRNGKey(s), num_paths=200_000,
                       smooth_fraction=0.002))
        for s in range(12)
    ])

    analytic = float(bachelier_price_single(F, K, T, SIG, R, True))

    standard_error = float(jnp.std(prices)) / math.sqrt(len(prices))

    assert abs(float(jnp.mean(prices)) - analytic) < 4.0 * standard_error


def test_basket_monte_carlo_matches_the_closed_form():
    model = BasketBachelierModel(scheme=EulerMaruyama())
    params = BasketBachelierParams(r=R, sigmas=SIGMAS, weights=W, corr=CORR)

    spots = jnp.array([95.0, 100.0, 110.0])

    prices = jnp.array([
        float(mc_price(model, params, spots, K, T, jax.random.PRNGKey(s),
                       num_paths=200_000, smooth_fraction=0.002))
        for s in range(12)
    ])

    analytic = float(basket_bachelier_price(spots, K, T, W, SIGMAS, CORR, R, True))

    standard_error = float(jnp.std(prices)) / math.sqrt(len(prices))

    assert abs(float(jnp.mean(prices)) - analytic) < 4.0 * standard_error


def test_euler_is_exact_for_bachelier():
    model = BachelierModel(scheme=EulerMaruyama())
    params = BachelierParams(sigma=SIG, r=R)

    def stepped(num_steps, seed):
        pricer = MonteCarloPricer(
            model,
            EuropeanCall(strike=K, smooth_fn=sigmoid_smooth, smooth_w=0.002 * SIG),
        )

        return float(pricer.price(
            s0=jnp.array([F]), params=params, maturity=T, num_paths=100_000,
            num_steps=num_steps, key=jax.random.PRNGKey(seed),
        )) * math.exp(-R * T)

    coarse = jnp.array([stepped(1, s) for s in range(8)])
    fine = jnp.array([stepped(120, s) for s in range(8)])

    spread = (float(jnp.std(coarse)) + float(jnp.std(fine))) / math.sqrt(8)

    assert abs(float(jnp.mean(coarse)) - float(jnp.mean(fine))) < 4.0 * spread


def test_antithetic_sampling_reduces_variance():
    model = BasketBachelierModel(scheme=EulerMaruyama())
    params = BasketBachelierParams(r=R, sigmas=SIGMAS, weights=W, corr=CORR)

    def spread(antithetic):
        return float(jnp.std(jnp.array([
            float(mc_price(model, params, jnp.full(N, 100.0), K, T,
                           jax.random.PRNGKey(s), num_paths=4_000,
                           antithetic=antithetic))
            for s in range(20)
        ])))

    assert spread(True) < spread(False)


def test_terminal_draws_reproduce_the_requested_correlation():
    model = BasketBachelierModel(scheme=EulerMaruyama())
    params = BasketBachelierParams(r=0.0, sigmas=jnp.full(N, 20.0),
                                   weights=W, corr=uniform_correlation(N, 0.5))

    blocks = model.terminal_state(jnp.full(N, 100.0), params, 1.0, 200_000,
                                  jax.random.PRNGKey(11), antithetic=False)

    realised = jnp.corrcoef(blocks[0].T)

    off_diagonal = realised[~jnp.eye(N, dtype=bool)]

    assert float(jnp.max(jnp.abs(off_diagonal - 0.5))) < 0.01


def test_basket_volatility_falls_as_correlation_falls():
    high = float(basket_normal_volatility(W, SIGMAS, uniform_correlation(N, 1.0)))
    low = float(basket_normal_volatility(W, SIGMAS, uniform_correlation(N, 0.0)))

    assert low < high

    assert abs(high - float(jnp.sum(W * SIGMAS))) < 1e-12


def test_symmetrize_makes_the_basket_price_permutation_invariant():
    model = BasketBachelierModel(scheme=EulerMaruyama())
    params = BasketBachelierParams(r=R, sigmas=jnp.full(N, 20.0), weights=W,
                                   corr=CORR)

    price_fn = make_feature_price(model, params, N, num_paths=20_000,
                                  symmetrize=True)

    key = jax.random.PRNGKey(3)

    values = [
        float(price_fn(jnp.array(list(spots) + [K, T]), key))
        for spots in [(90.0, 100.0, 115.0), (115.0, 90.0, 100.0), (100.0, 115.0, 90.0)]
    ]

    assert max(values) - min(values) == 0.0


def test_calibration_recovers_a_known_normal_volatility():
    truth = BachelierParams(sigma=17.5, r=0.03)

    strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 100.0])
    maturities = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 1.0])
    is_call = jnp.array([True] * 7)

    spot = 100.0

    synthetic = MarketData(
        spot=spot,
        strikes=strikes,
        maturities=maturities,
        market_prices=bachelier_price(truth, strikes, maturities, is_call, spot),
        is_call=is_call,
    )

    config = ExperimentConfig(data=DataConfig(pricing_model=BACHELIER))

    result = calibrate_bachelier(config, synthetic)

    assert result.converged
    assert abs(float(result.params.sigma) - 17.5) < 0.05
    assert abs(float(result.params.r) - 0.03) < 0.01

    assert "state" in result.assumptions

    assert result.diagnostics["residual_rmse"] < 0.05
    assert "residual_median_relative_pct" in result.diagnostics


def test_the_basket_declares_correlation_as_an_assumption():
    problem = _problem(BASKET_BACHELIER)

    assert problem.calibration.params.sigma == SIG


def test_the_bachelier_models_are_registered_alongside_the_others():
    assert {"bachelier", "basket_bachelier"} <= set(available_problems())


def test_bachelier_feature_layouts():
    assert _problem(BACHELIER).feature_names == ("S", "K", "T", "sigma", "r")
    assert _problem(BASKET_BACHELIER).feature_names == ("S1", "S2", "S3", "K", "T")

    wider = _problem(BASKET_BACHELIER, n_assets=5)

    assert wider.feature_names == ("S1", "S2", "S3", "S4", "S5", "K", "T")


def test_the_normal_domain_is_additive_not_multiplicative():
    problem = _problem(BACHELIER)

    low, high = problem.feature_bounds()

    spread = problem.data.domain_n_sigma * SIG * math.sqrt(
        problem.data.domain_horizon
    )

    assert abs(float(low[0]) - (100.0 - spread)) < 1e-9
    assert abs(float(high[0]) - (100.0 + spread)) < 1e-9

    assert float(low[3]) > 1.0


def test_both_bachelier_problems_offer_a_closed_form():
    for model in (BACHELIER, BASKET_BACHELIER):
        problem = _problem(model)

        assert problem.analytic_price(problem.baseline_features()) is not None


def test_the_basket_bachelier_reference_matches_its_own_closed_form():
    problem = _problem(BASKET_BACHELIER, sigma=20.0)

    x = problem.baseline_features()

    analytic = float(problem.analytic_price(x))

    draws = jnp.array([
        float(problem.reference_price(x, jax.random.PRNGKey(s))) for s in range(8)
    ])

    standard_error = float(jnp.std(draws)) / math.sqrt(len(draws))

    assert abs(float(jnp.mean(draws)) - analytic) < 6.0 * standard_error + 0.02


def test_baseline_and_reference_points_stay_in_domain():
    for model in (BACHELIER, BASKET_BACHELIER):
        problem = _problem(model)

        low, high = problem.feature_bounds()

        assert bool(jnp.all(problem.baseline_features() >= low - 1e-9)), model
        assert bool(jnp.all(problem.baseline_features() <= high + 1e-9)), model

        points = problem.reference_points(n_points=32)

        assert bool(jnp.all(points >= low - 1e-9)), model
        assert bool(jnp.all(points <= high + 1e-9)), model


def test_exposure_paths_carry_the_bachelier_feature_layout():
    for model in (BACHELIER, BASKET_BACHELIER):
        problem = _problem(model)

        time_grid, features = problem.exposure_paths(
            strike=100.0, horizon=1.0, num_paths=6, num_steps=8, min_maturity=0.05
        )

        assert features.shape == (6, 9, problem.n_features)

        names = problem.feature_names

        assert bool(jnp.all(features[:, :, names.index("K")] == 100.0)), model
        assert float(jnp.min(features[:, :, names.index("T")])) >= 0.05 - 1e-12


def test_payoff_switching_changes_the_scheme_and_the_bounds():
    european = _problem(BASKET_BACHELIER, payoff="european_call")
    asian = _problem(BASKET_BACHELIER, payoff="asian_call")

    assert european.arbitrage_bounds(european.baseline_features()) is not None
    assert asian.arbitrage_bounds(asian.baseline_features()) is None

    x = european.baseline_features()
    key = jax.random.PRNGKey(5)

    assert float(european.label_price_fn()(x, key)) == float(
        _problem(BASKET_BACHELIER, payoff="european_call").label_price_fn()(x, key)
    )


def test_put_prices_are_positive_and_parity_holds_through_the_problem():
    call = _problem(BACHELIER, payoff="european_call")
    put = _problem(BACHELIER, payoff="european_put")

    x = call.baseline_features()

    c = float(call.analytic_price(x))
    p = float(put.analytic_price(x))

    forward = float(bachelier_forward(x[0], x[2], x[4]))

    assert p > 0.0
    assert abs((c - p) - math.exp(-x[4] * x[2]) * (forward - x[1])) < 1e-9


def test_shape_constraints_follow_the_bachelier_conventions():
    call = {c.feature: c for c in _problem(BACHELIER).shape_constraints()}

    assert call["S"].low == 0.0 and call["S"].high == 1.0
    assert call["K"].high == 0.0
    assert call["sigma"].low == 0.0

    assert call["r"].low == 0.0

    basket = {c.feature: c for c in _problem(BASKET_BACHELIER).shape_constraints()}

    for i in range(3):
        assert abs(basket[f"S{i + 1}"].high - 1.0 / 3.0) < 1e-12


if __name__ == "__main__":
    for check in [
        test_put_call_parity,
        test_spot_quoting_carries_the_forward,
        test_price_is_increasing_in_volatility_and_bounded_below,
        test_a_normal_underlying_admits_negative_states,
        test_closed_form_greeks_match_autodiff,
        test_basket_greeks_match_autodiff_and_the_hessian_is_rank_one,
        test_basket_price_depends_on_the_spots_only_through_the_basket,
        test_feature_pricer_gradient_and_hvp_match_finite_differences,
        test_exact_sampling_is_unbiased_against_the_closed_form,
        test_basket_monte_carlo_matches_the_closed_form,
        test_euler_is_exact_for_bachelier,
        test_antithetic_sampling_reduces_variance,
        test_terminal_draws_reproduce_the_requested_correlation,
        test_basket_volatility_falls_as_correlation_falls,
        test_symmetrize_makes_the_basket_price_permutation_invariant,
        test_calibration_recovers_a_known_normal_volatility,
        test_the_basket_declares_correlation_as_an_assumption,
        test_the_bachelier_models_are_registered_alongside_the_others,
        test_bachelier_feature_layouts,
        test_the_normal_domain_is_additive_not_multiplicative,
        test_both_bachelier_problems_offer_a_closed_form,
        test_the_basket_bachelier_reference_matches_its_own_closed_form,
        test_baseline_and_reference_points_stay_in_domain,
        test_exposure_paths_carry_the_bachelier_feature_layout,
        test_payoff_switching_changes_the_scheme_and_the_bounds,
        test_put_prices_are_positive_and_parity_holds_through_the_problem,
        test_shape_constraints_follow_the_bachelier_conventions,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
