import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from kalibrierung.market_data import MarketData
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
    PricingProblem,
    ProblemSpec,
    available_problems,
    build_problem,
    register_problem,
)
from surrogate_modeling.validation import run_reference_validation


SPOT, SIGMA, R = 100.0, 0.2, 0.05


def _market_data():
    return MarketData(
        spot=SPOT,
        strikes=jnp.array([80.0, 100.0, 120.0]),
        maturities=jnp.array([0.25, 0.5, 1.0]),
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )


def _problem(pricing_model):
    config = ExperimentConfig(
        simulation=SimulationConfig(num_paths=4_000, num_steps=20, reference_paths=16_000),
        data=DataConfig(pricing_model=pricing_model),
        basket=BasketConfig(n_assets=3),
    )

    return build_problem(
        pricing_model,
        config=config,
        market_data=_market_data(),
        calibration=CalibrationResult(params=BlackScholesParams(r=R, sigma=SIGMA)),
    )


class _Surrogate:
    """Minimal stand-in with the batch interface the validation stage uses."""

    def __init__(self, fn):
        self.fn = fn

    def predict_price(self, x):
        return self.fn(x)

    def predict_prices(self, X):
        return jnp.array([float(self.fn(x)) for x in X])

    def predict_gradients(self, X):
        return jax.vmap(jax.grad(lambda x: jnp.asarray(self.fn(x), float)))(X)


class ToyProblem(PricingProblem):
    """Everything a new model must supply, and nothing more."""

    name = "toy"

    @property
    def feature_names(self):
        return ("a", "b")

    def sample_features(self, u):
        return u.at[:, 0].set(u[:, 0] * 10.0).at[:, 1].set(1.0 + 2.0 * u[:, 1])

    def label_price_fn(self):
        return lambda x, key: x[0] * x[1]


def test_a_minimal_problem_needs_only_the_four_required_members():
    problem = ToyProblem()

    low, high = problem.feature_bounds()

    assert bool(jnp.allclose(low, jnp.array([0.0, 1.0])))
    assert bool(jnp.allclose(high, jnp.array([10.0, 3.0])))

    assert bool(jnp.allclose(problem.baseline_features(), jnp.array([5.0, 2.0])))

    specs = problem.surface_specs()

    assert len(specs) == 1
    assert (specs[0].x_index, specs[0].y_index) == (0, 1)
    assert specs[0].y_range == (1.0, 3.0)

    assert problem.underlying_paths(problem.baseline_features()) is None
    assert (
        problem.reference_price(problem.baseline_features(), jax.random.PRNGKey(0))
        is None
    )
    assert problem.shape_constraints() == ()
    assert problem.analytic_price(problem.baseline_features()) is None
    assert problem.arbitrage_bounds(problem.baseline_features()) is None
    assert problem.exposure_paths(strike=1.0) is None
    assert problem.exposure_strikes() == {}

    assert problem.feature_labels == ("a", "b")
    assert problem.describe()["n_features"] == 2


def test_a_minimal_problem_can_be_trained_on():
    dataset = create_sobolev_dataset(ToyProblem(), sobolev_order=2, n_samples=5)

    assert dataset.X.shape == (5, 2)
    assert dataset.gradients.shape == (5, 2)
    assert dataset.hvps.shape == (5, 2)

    assert bool(jnp.allclose(dataset.gradients[:, 0], dataset.X[:, 1]))
    assert bool(jnp.allclose(dataset.gradients[:, 1], dataset.X[:, 0]))


def test_registering_a_problem_makes_it_selectable():
    register_problem(
        ProblemSpec(
            "toy_registered",
            build=lambda config, market_data, calibration: ToyProblem(),
            calibrate=lambda config, market_data: CalibrationResult(params=None),
        ),
        overwrite=True,
    )

    assert "toy_registered" in available_problems()

    config = ExperimentConfig(data=DataConfig(pricing_model="toy_registered"))

    problem = build_problem(
        config.data.pricing_model,
        config=config,
        market_data=None,
        calibration=None,
    )

    assert problem.feature_names == ("a", "b")


def test_registering_a_duplicate_name_is_refused():
    try:
        register_problem(ProblemSpec(BLACK_SCHOLES, lambda **k: None, lambda **k: None))
    except ValueError as e:
        assert "already registered" in str(e)
    else:
        assert False, "silently replacing a registered problem should raise"


def test_validation_reports_a_perfect_surrogate_as_accurate():
    problem = _problem(BLACK_SCHOLES)

    surrogate = _Surrogate(problem.analytic_price)

    summary = run_reference_validation(surrogate, problem, n_points=32)

    assert summary["surrogate_median_relative_pct"] < 5.0
    assert abs(summary["surrogate_bias"]) < 1.0
    assert "reference_vs_analytic_bias" in summary
    assert summary["negative_prices"] == 0.0


def test_validation_runs_for_a_model_without_a_closed_form():
    problem = _problem(BASKET_BLACK_SCHOLES)

    surrogate = _Surrogate(
        lambda x: problem.reference_price(x, jax.random.PRNGKey(4))
    )

    summary = run_reference_validation(surrogate, problem, n_points=16)

    assert "surrogate_bias" in summary
    assert "reference_vs_analytic_bias" not in summary
    assert summary["surrogate_median_relative_pct"] < 25.0


def test_validation_catches_an_arbitrage_violating_surrogate():
    problem = _problem(BLACK_SCHOLES)

    summary = run_reference_validation(
        _Surrogate(lambda x: jnp.asarray(1e6) + 0.0 * jnp.sum(x)),
        problem,
        n_points=16,
    )

    assert summary["arbitrage_violation_pct"] == 100.0
    assert summary["arbitrage_worst_breach"] > 0.0


def test_validation_catches_a_surrogate_that_broke_exchangeability():
    problem = _problem(BASKET_BLACK_SCHOLES)

    asymmetric = _Surrogate(lambda x: 3.0 * x[0] + x[1] + x[2])
    symmetric = _Surrogate(lambda x: x[0] + x[1] + x[2])

    broken = run_reference_validation(asymmetric, problem, n_points=16)
    intact = run_reference_validation(symmetric, problem, n_points=16)

    assert broken["permutation_worst_deviation"] > 1.0
    assert intact["permutation_worst_deviation"] < 1e-9


def test_validation_skips_exchangeability_for_a_single_asset_model():
    summary = run_reference_validation(
        _Surrogate(_problem(BLACK_SCHOLES).analytic_price),
        _problem(BLACK_SCHOLES),
        n_points=8,
    )

    assert "permutation_worst_deviation" not in summary


def test_validation_says_so_when_no_benchmark_exists(capsys=None):
    summary = run_reference_validation(
        _Surrogate(lambda x: x[0] * x[1]), ToyProblem(), n_points=3
    )

    assert summary == {}


if __name__ == "__main__":
    for check in [
        test_a_minimal_problem_needs_only_the_four_required_members,
        test_a_minimal_problem_can_be_trained_on,
        test_registering_a_problem_makes_it_selectable,
        test_registering_a_duplicate_name_is_refused,
        test_validation_reports_a_perfect_surrogate_as_accurate,
        test_validation_runs_for_a_model_without_a_closed_form,
        test_validation_catches_an_arbitrage_violating_surrogate,
        test_validation_catches_a_surrogate_that_broke_exchangeability,
        test_validation_skips_exchangeability_for_a_single_asset_model,
        test_validation_says_so_when_no_benchmark_exists,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
