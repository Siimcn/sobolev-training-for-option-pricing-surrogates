import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
from marktsimulation.basket_mc import (
    is_exchangeable,
    make_basket_feature_price,
    uniform_correlation,
)
from marktsimulation.black_scholes import black_scholes_price_single
from marktsimulation.pricing_model import BlackScholesParams
from pipeline.config import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
    BasketConfig,
    DataConfig,
    ExperimentConfig,
    SimulationConfig,
)
from surrogate_modeling.data_generation import create_sobolev_dataset
from surrogate_modeling.pricing_problem import (
    CalibrationResult,
    available_problems,
    build_problem,
)


SPOT, SIGMA, R = 100.0, 0.2, 0.05


def _market_data():
    strikes = jnp.array([80.0, 100.0, 120.0])
    maturities = jnp.array([0.25, 0.5, 1.0])

    return MarketData(
        spot=SPOT,
        strikes=strikes,
        maturities=maturities,
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )


def _params():
    return BlackScholesParams(r=R, sigma=SIGMA)


def _config(pricing_model, basket=None, min_maturity=None, num_paths=4_000):
    return ExperimentConfig(
        simulation=SimulationConfig(
            num_paths=num_paths, num_steps=20, reference_paths=4 * num_paths
        ),
        data=DataConfig(pricing_model=pricing_model, min_maturity=min_maturity),
        basket=basket or BasketConfig(n_assets=3),
    )


def _problem(pricing_model, basket=None, min_maturity=None, num_paths=4_000):
    config = _config(pricing_model, basket, min_maturity, num_paths)

    return build_problem(
        pricing_model,
        config=config,
        market_data=_market_data(),
        calibration=CalibrationResult(params=_params()),
    )


def _dataset(pricing_model, basket=None, n_samples=3, sobolev_order=2, min_maturity=None):
    return create_sobolev_dataset(
        _problem(pricing_model, basket, min_maturity),
        sobolev_order,
        n_samples=n_samples,
    )


def test_configured_default_is_a_known_model():
    assert ExperimentConfig().data.pricing_model in available_problems()


def test_both_built_in_models_are_registered():
    assert BLACK_SCHOLES in available_problems()
    assert BASKET_BLACK_SCHOLES in available_problems()


def test_black_scholes_feature_layout():
    assert _problem(BLACK_SCHOLES).feature_names == ("S", "K", "T", "sigma", "r")


def test_basket_feature_layout_follows_the_asset_count():
    assert _problem(BASKET_BLACK_SCHOLES).feature_names == ("S1", "S2", "S3", "K", "T")

    wider = _problem(
        BASKET_BLACK_SCHOLES,
        basket=BasketConfig(n_assets=5),
    )

    assert wider.feature_names == ("S1", "S2", "S3", "S4", "S5", "K", "T")


def test_weights_default_to_equal_shares():
    equal = _problem(
        BASKET_BLACK_SCHOLES, basket=BasketConfig(n_assets=4)
    )

    assert bool(jnp.allclose(equal.weights, 0.25))

    custom = _problem(
        BASKET_BLACK_SCHOLES,
        basket=BasketConfig(n_assets=2, weights=(0.3, 0.7), symmetrize=False),
    )

    assert bool(jnp.allclose(custom.weights, jnp.array([0.3, 0.7])))


def test_unknown_pricing_model_is_rejected():
    try:
        ExperimentConfig(data=DataConfig(pricing_model="sabr"))
    except ValueError as e:
        assert "sabr" in str(e)
    else:
        assert False, "an unknown pricing model should raise"


def test_unknown_pricing_model_rejected_by_the_registry():
    try:
        build_problem("sabr", config=None, market_data=None, calibration=None)
    except ValueError as e:
        assert "sabr" in str(e)
    else:
        assert False, "an unknown pricing model should raise"


def test_weight_length_must_match_asset_count():
    try:
        ExperimentConfig(basket=BasketConfig(n_assets=3, weights=(0.5, 0.5)))
    except ValueError:
        pass
    else:
        assert False, "mismatched weights should raise"


def test_network_input_width_is_derived_by_default():
    assert ExperimentConfig().network.in_size is None


def test_baseline_point_lies_inside_the_training_domain():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model, min_maturity=0.3)

        low, high = problem.feature_bounds()
        baseline = problem.baseline_features()

        assert baseline.shape == (problem.n_features,)
        assert bool(jnp.all(baseline >= low - 1e-9)), model
        assert bool(jnp.all(baseline <= high + 1e-9)), model


def test_surface_specs_stay_inside_the_sampled_domain():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model, min_maturity=0.3)

        low, high = problem.feature_bounds()

        specs = problem.surface_specs()

        assert len(specs) == problem.n_features - 1

        for spec in specs:
            assert spec.x_index != spec.y_index

            for index, (lo, hi) in [
                (spec.x_index, spec.x_range),
                (spec.y_index, spec.y_range),
            ]:
                assert lo >= float(low[index]) - 1e-9
                assert hi <= float(high[index]) + 1e-9


def test_feature_bounds_match_the_drawn_samples():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model)

        low, high = problem.feature_bounds()

        u = jax.random.uniform(
            jax.random.PRNGKey(3), shape=(64, problem.n_features)
        )

        X = problem.sample_features(u)

        assert bool(jnp.all(X >= low - 1e-9)), model
        assert bool(jnp.all(X <= high + 1e-9)), model


def test_feature_labels_cover_every_feature():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model)

        assert len(problem.feature_labels) == problem.n_features
        assert len(problem.feature_names) == problem.n_features


def test_only_the_basket_declares_exchangeable_features():
    assert _problem(BLACK_SCHOLES).exchangeable_features == ()
    assert _problem(BASKET_BLACK_SCHOLES).exchangeable_features == (0, 1, 2)

    uneven = _problem(
        BASKET_BLACK_SCHOLES,
        basket=BasketConfig(n_assets=3, weights=(0.5, 0.3, 0.2), symmetrize=False),
    )

    assert uneven.exchangeable_features == ()


def test_single_asset_basket_matches_the_analytic_call():
    price_fn = make_basket_feature_price(
        weights=jnp.array([1.0]),
        corr=uniform_correlation(1, 0.0),
        sigmas=jnp.array([SIGMA]),
        r=R,
        num_paths=60_000,
        num_steps=50,
    )

    price = float(price_fn(jnp.array([SPOT, 100.0, 1.0]), jax.random.PRNGKey(0)))
    analytic = float(black_scholes_price_single(SPOT, 100.0, 1.0, SIGMA, R))

    assert abs(price - analytic) < 0.5


def test_basket_price_is_differentiable_twice():
    price_fn = make_basket_feature_price(
        weights=jnp.full(3, 1.0 / 3.0),
        corr=uniform_correlation(3, 0.5),
        sigmas=jnp.full(3, SIGMA),
        r=R,
        num_paths=8_000,
        num_steps=20,
    )

    x = jnp.array([100.0, 100.0, 100.0, 100.0, 1.0])

    keyed = lambda z: price_fn(z, jax.random.PRNGKey(0))

    gradient = jax.grad(keyed)(x)
    hvp = jax.jvp(jax.grad(keyed), (x,), (jnp.ones(5),))[1]

    assert gradient.shape == (5,)
    assert hvp.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(hvp)))

    assert bool(jnp.all(gradient[:3] > 0.0))
    assert bool(jnp.all(gradient[:3] < 1.0 / 3.0))

    assert float(gradient[3]) < 0.0


def test_european_payoff_is_priced_from_an_exact_terminal_draw():
    def priced(num_steps):
        return float(
            make_basket_feature_price(
                weights=jnp.full(3, 1.0 / 3.0),
                corr=uniform_correlation(3, 0.5),
                sigmas=jnp.full(3, SIGMA),
                r=R,
                payoff="european_call",
                num_paths=20_000,
                num_steps=num_steps,
            )(jnp.array([100.0, 100.0, 100.0, 100.0, 1.0]), jax.random.PRNGKey(0))
        )

    assert priced(5) == priced(500)


def test_path_dependent_payoff_still_uses_the_stepping_scheme():
    def priced(num_steps):
        return float(
            make_basket_feature_price(
                weights=jnp.full(3, 1.0 / 3.0),
                corr=uniform_correlation(3, 0.5),
                sigmas=jnp.full(3, SIGMA),
                r=R,
                payoff="asian_call",
                num_paths=8_000,
                num_steps=num_steps,
            )(jnp.array([100.0, 100.0, 100.0, 100.0, 1.0]), jax.random.PRNGKey(0))
        )

    assert priced(5) != priced(50)


def test_exact_sampling_is_unbiased_against_the_closed_form():
    price_fn = make_basket_feature_price(
        weights=jnp.array([1.0]),
        corr=uniform_correlation(1, 0.0),
        sigmas=jnp.array([SIGMA]),
        r=R,
        num_paths=20_000,
    )

    x = jnp.array([SPOT, 100.0, 1.0])

    prices = jnp.array(
        [float(price_fn(x, jax.random.PRNGKey(s))) for s in range(16)]
    )

    analytic = float(black_scholes_price_single(SPOT, 100.0, 1.0, SIGMA, R))

    relative = float(jnp.mean(prices)) / analytic - 1.0

    assert -0.01 < relative <= 0.001, f"unexpected sampling bias {relative:+.4%}"


def test_exact_sampling_stays_twice_differentiable():
    price_fn = make_basket_feature_price(
        weights=jnp.full(3, 1.0 / 3.0),
        corr=uniform_correlation(3, 0.5),
        sigmas=jnp.full(3, SIGMA),
        r=R,
        num_paths=8_000,
        symmetrize=True,
    )

    x = jnp.array([100.0, 100.0, 100.0, 100.0, 1.0])
    keyed = lambda z: price_fn(z, jax.random.PRNGKey(0))

    gradient = jax.grad(keyed)(x)
    hvp = jax.jvp(jax.grad(keyed), (x,), (jnp.ones(5),))[1]

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(hvp)))

    assert bool(jnp.all((gradient[:3] > 0.0) & (gradient[:3] < 1.0 / 3.0)))
    assert float(gradient[3]) < 0.0
    assert float(gradient[4]) > 0.0


def test_uniform_correlation_matrix():
    corr = uniform_correlation(3, 0.5)

    assert corr.shape == (3, 3)
    assert bool(jnp.all(jnp.diag(corr) == 1.0))
    assert float(corr[0, 1]) == 0.5
    assert bool(jnp.all(corr == corr.T))


def test_black_scholes_dataset_shapes():
    dataset = _dataset(BLACK_SCHOLES, n_samples=3)

    assert dataset.X.shape == (3, 5)
    assert dataset.y.shape == (3,)
    assert dataset.gradients.shape == (3, 5)
    assert dataset.hvps.shape == (3, 5)
    assert dataset.V.shape == (3, 5)


def test_basket_dataset_shapes_follow_the_asset_count():
    dataset = _dataset(BASKET_BLACK_SCHOLES, n_samples=3)

    assert dataset.X.shape == (3, 5)
    assert dataset.gradients.shape == (3, 5)
    assert dataset.input_dim == 5

    wider = _dataset(
        BASKET_BLACK_SCHOLES,
        basket=BasketConfig(n_assets=4),
        n_samples=2,
    )

    assert wider.X.shape == (2, 6)
    assert wider.input_dim == 6


def test_basket_domain_respects_the_market_ranges():
    dataset = _dataset(BASKET_BLACK_SCHOLES, n_samples=8)

    strikes, maturities = dataset.X[:, 3], dataset.X[:, 4]

    assert bool(jnp.all(strikes >= 80.0)) and bool(jnp.all(strikes <= 120.0))
    assert bool(jnp.all(maturities >= 0.25)) and bool(jnp.all(maturities <= 1.0))

    assert bool(jnp.all(dataset.X[:, :3] > 0.0))


def test_sobolev_order_one_skips_the_hvps():
    dataset = _dataset(BLACK_SCHOLES, n_samples=2, sobolev_order=1)

    assert dataset.hvps is None
    assert dataset.gradients is not None


def test_preview_paths_read_the_right_feature_layout():
    for model, x, horizon in [
        (
            BLACK_SCHOLES,
            jnp.array([312.34, 300.0, 0.83, 0.2883, 0.0414]),
            0.83,
        ),
        (
            BASKET_BLACK_SCHOLES,
            jnp.array([180.73, 695.27, 579.35, 362.39, 0.83]),
            0.83,
        ),
    ]:
        time_grid, paths = _problem(model).underlying_paths(x, num_paths=8)

        assert abs(float(time_grid[-1]) - horizon) < 1e-9
        assert bool(jnp.all(jnp.isfinite(paths)))
        assert bool(jnp.all(paths > 0.0))
        assert float(jnp.max(paths)) < 1e5


def test_basket_preview_paths_start_at_the_weighted_basket():
    x = jnp.array([180.73, 695.27, 579.35, 362.39, 0.83])

    _, paths = _problem(BASKET_BLACK_SCHOLES).underlying_paths(x, num_paths=8)

    assert abs(float(paths[0, 0]) - float(jnp.mean(x[:3]))) < 1e-9


def test_exposure_paths_carry_the_problems_own_feature_layout():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model)

        time_grid, features = problem.exposure_paths(
            strike=110.0, horizon=1.0, num_paths=6, num_steps=8
        )

        assert features.shape == (6, 9, problem.n_features)
        assert time_grid.shape == (9,)
        assert bool(jnp.all(jnp.isfinite(features)))

        names = problem.feature_names

        strike_column = features[:, :, names.index("K")]
        maturity_column = features[:, :, names.index("T")]

        assert bool(jnp.all(strike_column == 110.0)), model

        assert abs(float(maturity_column[0, 0]) - 1.0) < 1e-9
        assert float(maturity_column[0, -1]) < 1e-6

        assert float(jnp.std(features[:, -1, 0])) > 0.0


def test_only_black_scholes_offers_a_closed_form():
    bs = _problem(BLACK_SCHOLES)
    basket = _problem(BASKET_BLACK_SCHOLES)

    x_bs = jnp.array([SPOT, 100.0, 0.5, SIGMA, R])

    analytic = bs.analytic_price(x_bs)

    assert analytic is not None
    assert abs(
        float(analytic) - float(black_scholes_price_single(SPOT, 100.0, 0.5, SIGMA, R))
    ) < 1e-10

    assert basket.analytic_price(basket.baseline_features()) is None


def test_arbitrage_bounds_bracket_the_true_price():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model)

        x = problem.baseline_features()

        lower, upper = problem.arbitrage_bounds(x)

        price = float(problem.reference_price(x, jax.random.PRNGKey(7)))

        assert lower - 1e-6 <= price <= upper + 1e-6, model


def test_reference_price_uses_the_key_it_is_given():
    problem = _problem(BASKET_BLACK_SCHOLES)

    x = problem.baseline_features()

    a = float(problem.reference_price(x, jax.random.PRNGKey(1)))
    b = float(problem.reference_price(x, jax.random.PRNGKey(2)))

    assert a != b
    assert abs(a - b) / a < 0.1


def test_reference_points_are_in_domain():
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model, min_maturity=0.3)

        low, high = problem.feature_bounds()

        points = problem.reference_points(n_points=12)

        assert points.shape == (12, problem.n_features)
        assert bool(jnp.all(points >= low - 1e-9)), model
        assert bool(jnp.all(points <= high + 1e-9)), model


def _basket_pricer(symmetrize, n=3, rho=0.5, weights=None):
    w = jnp.full(n, 1.0 / n) if weights is None else jnp.asarray(weights)

    priced = make_basket_feature_price(
        weights=w,
        corr=uniform_correlation(n, rho),
        sigmas=jnp.full(n, SIGMA),
        r=R,
        num_paths=6_000,
        num_steps=20,
        symmetrize=symmetrize,
    )

    return lambda x: priced(x, jax.random.PRNGKey(0))


def test_raw_estimator_is_not_permutation_invariant():
    price_fn = _basket_pricer(symmetrize=False)

    values = [
        float(price_fn(jnp.array(list(spots) + [110.0, 0.8])))
        for spots in [(80.0, 100.0, 130.0), (130.0, 80.0, 100.0), (100.0, 130.0, 80.0)]
    ]

    assert max(values) - min(values) > 1e-6


def test_symmetrize_makes_the_price_permutation_invariant():
    price_fn = _basket_pricer(symmetrize=True)

    values = [
        float(price_fn(jnp.array(list(spots) + [110.0, 0.8])))
        for spots in [(80.0, 100.0, 130.0), (130.0, 80.0, 100.0), (100.0, 130.0, 80.0)]
    ]

    assert max(values) - min(values) == 0.0


def test_symmetrized_gradient_permutes_with_the_input():
    price_fn = _basket_pricer(symmetrize=True)

    g = jax.grad(price_fn)(jnp.array([80.0, 100.0, 130.0, 110.0, 0.8]))
    g_rolled = jax.grad(price_fn)(jnp.array([130.0, 80.0, 100.0, 110.0, 0.8]))

    assert bool(jnp.allclose(jnp.array([g_rolled[1], g_rolled[2], g_rolled[0]]), g[:3]))
    assert bool(jnp.all(g[:3] > 0.0))


def test_symmetrized_price_stays_twice_differentiable():
    price_fn = _basket_pricer(symmetrize=True)

    x = jnp.array([80.0, 100.0, 130.0, 110.0, 0.8])
    hvp = jax.jvp(jax.grad(price_fn), (x,), (jnp.ones(5),))[1]

    assert bool(jnp.all(jnp.isfinite(hvp)))


def test_exchangeability_check():
    w, s = jnp.full(3, 1 / 3), jnp.full(3, SIGMA)

    assert is_exchangeable(w, s, uniform_correlation(3, 0.5))
    assert not is_exchangeable(jnp.array([0.5, 0.3, 0.2]), s, uniform_correlation(3, 0.5))
    assert not is_exchangeable(w, jnp.array([0.1, 0.2, 0.3]), uniform_correlation(3, 0.5))


def test_symmetrize_rejects_a_non_exchangeable_basket():
    try:
        _basket_pricer(symmetrize=True, weights=(0.5, 0.3, 0.2))
    except ValueError as e:
        assert "exchangeable" in str(e)
    else:
        assert False, "unequal weights with symmetrize should raise"


def test_min_maturity_floors_the_sampled_maturities():
    floored = _problem(BASKET_BLACK_SCHOLES, min_maturity=0.4)
    unfloored = _problem(BASKET_BLACK_SCHOLES)

    t = floored.feature_names.index("T")

    assert abs(float(floored.feature_bounds()[0][t]) - 0.4) < 1e-12
    assert abs(float(unfloored.feature_bounds()[0][t]) - 0.25) < 1e-12

    labels = _dataset(
        BASKET_BLACK_SCHOLES, n_samples=16, sobolev_order=1, min_maturity=0.4
    )

    assert bool(jnp.all(labels.X[:, t] >= 0.4))


def test_min_maturity_above_the_longest_expiry_raises():
    try:
        _dataset(BLACK_SCHOLES, n_samples=2, sobolev_order=1, min_maturity=5.0)
    except ValueError as e:
        assert "min_maturity" in str(e)
    else:
        assert False, "an unreachable maturity floor should raise"


if __name__ == "__main__":
    for check in [
        test_configured_default_is_a_known_model,
        test_both_built_in_models_are_registered,
        test_black_scholes_feature_layout,
        test_basket_feature_layout_follows_the_asset_count,
        test_weights_default_to_equal_shares,
        test_unknown_pricing_model_is_rejected,
        test_unknown_pricing_model_rejected_by_the_registry,
        test_weight_length_must_match_asset_count,
        test_network_input_width_is_derived_by_default,
        test_baseline_point_lies_inside_the_training_domain,
        test_surface_specs_stay_inside_the_sampled_domain,
        test_feature_bounds_match_the_drawn_samples,
        test_feature_labels_cover_every_feature,
        test_only_the_basket_declares_exchangeable_features,
        test_single_asset_basket_matches_the_analytic_call,
        test_basket_price_is_differentiable_twice,
        test_european_payoff_is_priced_from_an_exact_terminal_draw,
        test_path_dependent_payoff_still_uses_the_stepping_scheme,
        test_exact_sampling_is_unbiased_against_the_closed_form,
        test_exact_sampling_stays_twice_differentiable,
        test_uniform_correlation_matrix,
        test_black_scholes_dataset_shapes,
        test_basket_dataset_shapes_follow_the_asset_count,
        test_basket_domain_respects_the_market_ranges,
        test_sobolev_order_one_skips_the_hvps,
        test_preview_paths_read_the_right_feature_layout,
        test_basket_preview_paths_start_at_the_weighted_basket,
        test_exposure_paths_carry_the_problems_own_feature_layout,
        test_only_black_scholes_offers_a_closed_form,
        test_arbitrage_bounds_bracket_the_true_price,
        test_reference_price_uses_the_key_it_is_given,
        test_reference_points_are_in_domain,
        test_raw_estimator_is_not_permutation_invariant,
        test_symmetrize_makes_the_price_permutation_invariant,
        test_symmetrized_gradient_permutes_with_the_input,
        test_symmetrized_price_stays_twice_differentiable,
        test_exchangeability_check,
        test_symmetrize_rejects_a_non_exchangeable_basket,
        test_min_maturity_floors_the_sampled_maturities,
        test_min_maturity_above_the_longest_expiry_raises,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
