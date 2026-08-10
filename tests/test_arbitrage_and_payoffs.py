import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
from marktsimulation.payoff import (
    available_payoffs,
    build_payoff,
    payoff_spec,
    register_payoff,
    PayoffSpec,
)
from marktsimulation.pricing_model import BlackScholesParams
from pipeline.config import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
    BasketConfig,
    DataConfig,
    ExperimentConfig,
    PayoffConfig,
    SimulationConfig,
)
from surrogate_modeling.pricing_problem import CalibrationResult, build_problem
from surrogate_modeling.surrogate_model import SurrogateModel


SPOT, SIGMA, R = 100.0, 0.2, 0.05


def _market_data():
    return MarketData(
        spot=SPOT,
        strikes=jnp.array([80.0, 100.0, 120.0]),
        maturities=jnp.array([0.25, 0.5, 1.0]),
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )


def _problem(model=BASKET_BLACK_SCHOLES, payoff="european_call", min_maturity=0.05):
    config = ExperimentConfig(
        payoff=PayoffConfig(name=payoff),
        simulation=SimulationConfig(num_paths=2_000, num_steps=10),
        data=DataConfig(pricing_model=model, min_maturity=min_maturity),
        basket=BasketConfig(n_assets=3),
    )

    return build_problem(
        model,
        config=config,
        market_data=_market_data(),
        calibration=CalibrationResult(params=BlackScholesParams(r=R, sigma=SIGMA)),
    )


# ------------------------------------------------------------- positivity


def _network(constant):
    return lambda x: jnp.asarray(constant) + 0.0 * jnp.sum(x)


def test_the_surrogate_output_is_left_unconstrained():
    """
    A softplus positivity transform was tried and removed.

    Any map from R onto (0, inf) that behaves like the identity for large
    values must have a vanishing derivative for very negative ones, and the
    two are rigidly linked: a floor below 0.01 forces saturation at about
    -10 price units, which an untrained network reaches easily. In a
    600-sample run the gradient underflowed to exactly zero and training
    froze at epoch 25 with a bit-identical loss.

    Negative prices are reported by the validation stage instead, and the
    exposure profile no longer leaves the training domain - which is where
    they actually came from.
    """

    model = SurrogateModel(_network(-5.0), y_std=jnp.array(100.0))

    assert float(model.predict_price(jnp.zeros(5))) < 0.0

    # the gradient stays alive everywhere, which is what the transform broke
    scaled = SurrogateModel(lambda x: jnp.sum(x), y_std=jnp.array(100.0))

    for level in (-1e4, -1.0, 0.0, 1e4):
        gradient = scaled.predict_gradient(jnp.full(5, level))

        assert bool(jnp.all(jnp.abs(gradient) > 0.0)), level


# ------------------------------------------------------- exposure domain


def test_exposure_paths_stop_at_the_training_floor():
    # regression: the profile ran the remaining maturity down to 1e-8,
    # evaluating the surrogate below min_maturity where it was never fitted
    for model in (BLACK_SCHOLES, BASKET_BLACK_SCHOLES):
        problem = _problem(model)

        _, features = problem.exposure_paths(
            strike=SPOT, horizon=1.0, num_paths=4, num_steps=20, min_maturity=0.05
        )

        maturity = features[:, :, problem.feature_names.index("T")]

        assert float(jnp.min(maturity)) >= 0.05 - 1e-12, model
        assert abs(float(jnp.max(maturity)) - 1.0) < 1e-9


def test_exposure_paths_without_a_floor_still_reach_expiry():
    problem = _problem(BLACK_SCHOLES)

    _, features = problem.exposure_paths(
        strike=SPOT, horizon=1.0, num_paths=4, num_steps=20
    )

    assert float(jnp.min(features[:, :, 2])) < 1e-6


# ------------------------------------------------------- arbitrage bounds


def test_call_and_put_bounds_have_the_right_shape():
    call = _problem(BLACK_SCHOLES, payoff="european_call")
    put = _problem(BLACK_SCHOLES, payoff="european_put")

    x = jnp.array([SPOT, 120.0, 1.0, SIGMA, R])

    call_low, call_high = call.arbitrage_bounds(x)
    put_low, put_high = put.arbitrage_bounds(x)

    assert call_low == 0.0 and abs(call_high - SPOT) < 1e-9
    assert put_low > 0.0 and put_high > put_low


def test_a_path_dependent_payoff_declares_no_bounds():
    # an arithmetic average is less volatile than the terminal value, so
    # the European bounds do not carry over
    asian = _problem(BASKET_BLACK_SCHOLES, payoff="asian_call")

    assert asian.arbitrage_bounds(asian.baseline_features()) is None


def test_shape_constraints_follow_the_payoff():
    call = _problem(BLACK_SCHOLES, payoff="european_call")
    put = _problem(BLACK_SCHOLES, payoff="european_put")

    call_by_feature = {c.feature: c for c in call.shape_constraints()}
    put_by_feature = {c.feature: c for c in put.shape_constraints()}

    assert call_by_feature["S"].low == 0.0 and call_by_feature["S"].high == 1.0
    assert put_by_feature["S"].low == -1.0 and put_by_feature["S"].high == 0.0

    assert call_by_feature["K"].high == 0.0
    assert put_by_feature["K"].low == 0.0


def test_basket_deltas_are_bounded_by_their_weight():
    problem = _problem(BASKET_BLACK_SCHOLES)

    constraints = {c.feature: c for c in problem.shape_constraints()}

    for i in range(3):
        assert abs(constraints[f"S{i + 1}"].high - 1.0 / 3.0) < 1e-12


# ---------------------------------------------------------- payoff registry


def test_built_in_payoffs_are_registered():
    assert set(available_payoffs()) >= {
        "european_call",
        "european_put",
        "asian_call",
        "asian_put",
    }


def test_only_path_dependent_payoffs_need_the_stepping_scheme():
    assert payoff_spec("european_call").path_dependent is False
    assert payoff_spec("asian_call").path_dependent is True


def test_an_unknown_payoff_is_rejected_by_the_config():
    try:
        ExperimentConfig(payoff=PayoffConfig(name="lookback_call"))
    except ValueError as e:
        assert "lookback_call" in str(e)
    else:
        assert False, "an unregistered payoff must raise"


def test_a_new_payoff_needs_only_a_registration():
    from marktsimulation.payoff import EuropeanPayoff, sigmoid_smooth

    def straddle(strike, smooth_fn=sigmoid_smooth, smooth_w=0.05):
        return EuropeanPayoff(
            strike=strike, omega=1.0, smooth_fn=smooth_fn, smooth_w=smooth_w
        )

    register_payoff(
        PayoffSpec("test_straddle", straddle, path_dependent=False), overwrite=True
    )

    assert "test_straddle" in available_payoffs()

    payoff = build_payoff("test_straddle", strike=100.0, smooth_w=1.0)

    assert float(payoff(jnp.array(120.0))) > 0.0

    # and it is selectable without touching anything else
    config = ExperimentConfig(payoff=PayoffConfig(name="test_straddle"))

    assert config.payoff.name == "test_straddle"


def test_registering_a_duplicate_payoff_is_refused():
    try:
        register_payoff(payoff_spec("european_call"))
    except ValueError as e:
        assert "already registered" in str(e)
    else:
        assert False, "silently replacing a payoff must raise"


if __name__ == "__main__":
    for check in [
        test_the_surrogate_output_is_left_unconstrained,
        test_exposure_paths_stop_at_the_training_floor,
        test_exposure_paths_without_a_floor_still_reach_expiry,
        test_call_and_put_bounds_have_the_right_shape,
        test_a_path_dependent_payoff_declares_no_bounds,
        test_shape_constraints_follow_the_payoff,
        test_basket_deltas_are_bounded_by_their_weight,
        test_built_in_payoffs_are_registered,
        test_only_path_dependent_payoffs_need_the_stepping_scheme,
        test_an_unknown_payoff_is_rejected_by_the_config,
        test_a_new_payoff_needs_only_a_registration,
        test_registering_a_duplicate_payoff_is_refused,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
