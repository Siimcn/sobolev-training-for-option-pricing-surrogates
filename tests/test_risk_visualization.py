import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from market_simulation.black_scholes import black_scholes_price_single
from risk_visualization.riskengine import RiskEngine
from risk_visualization.visualizer import Visualizer
from surrogate_modeling.surrogate_model import SurrogateModel

N_PATHS, N_STEPS = 8, 5
SPOT, STRIKE, SIGMA, R = 100.0, 100.0, 0.2, 0.05


class ConstantSurrogate:
    """Fixed price for every state, so XVA can be checked in closed form."""

    def __init__(self, value):
        self.value = value

    def predict_price(self, x):
        return jnp.asarray(self.value) + 0.0 * jnp.sum(x)


class TempLogger:
    def __init__(self, directory):
        self.directory = directory

    def path(self, filename):
        return os.path.join(self.directory, filename)


def _feature_paths(time_grid):
    remaining = jnp.maximum(1.0 - time_grid, 1e-8)

    S = jnp.full((N_PATHS, N_STEPS), SPOT)

    return jnp.stack(
        [
            S,
            jnp.full_like(S, STRIKE),
            jnp.broadcast_to(remaining, S.shape),
            jnp.full_like(S, SIGMA),
            jnp.full_like(S, R),
        ],
        axis=-1,
    )


def _expected_xva(engine, exposure, time_grid, rate):
    discounted = exposure * jnp.exp(-rate * time_grid)

    survival = jnp.exp(-engine.hazard_rate * time_grid)
    default_probs = jnp.concatenate([survival[:-1] - survival[1:], jnp.array([0.0])])

    lgd = 1.0 - engine.recovery_rate

    return float(lgd * jnp.sum(jnp.maximum(discounted, 0.0) * default_probs))


def test_positive_exposure_produces_cva_and_no_dva():
    engine = RiskEngine()
    time_grid = jnp.linspace(0.0, 1.0, N_STEPS)

    result = engine.compute_xva_risk(
        ConstantSurrogate(7.0), _feature_paths(time_grid), time_grid, R
    )

    assert abs(result["CVA"] - _expected_xva(engine, 7.0, time_grid, R)) < 1e-10
    assert result["DVA"] == 0.0
    assert abs(result["NetXVA"] - result["CVA"]) < 1e-12
    assert result["V_matrix"].shape == (N_PATHS, N_STEPS)


def test_negative_exposure_produces_dva_and_no_cva():
    engine = RiskEngine()
    time_grid = jnp.linspace(0.0, 1.0, N_STEPS)

    result = engine.compute_xva_risk(
        ConstantSurrogate(-7.0), _feature_paths(time_grid), time_grid, R
    )

    assert result["CVA"] == 0.0
    assert abs(result["DVA"] - _expected_xva(engine, 7.0, time_grid, R)) < 1e-10
    assert abs(result["NetXVA"] + result["DVA"]) < 1e-12


def test_exposures_are_non_negative_and_net_is_consistent():
    engine = RiskEngine()
    time_grid = jnp.linspace(0.0, 1.0, N_STEPS)

    result = engine.compute_xva_risk(
        ConstantSurrogate(3.0), _feature_paths(time_grid), time_grid, R
    )

    assert bool(jnp.all(result["EPE"] >= 0.0))
    assert bool(jnp.all(result["ENE"] >= 0.0))
    assert abs(result["NetXVA"] - (result["CVA"] - result["DVA"])) < 1e-12


def test_recovery_and_hazard_rates_scale_the_adjustment():
    time_grid = jnp.linspace(0.0, 1.0, N_STEPS)
    paths = _feature_paths(time_grid)

    base = RiskEngine(recovery_rate=0.4, hazard_rate=0.02)
    no_recovery = RiskEngine(recovery_rate=0.0, hazard_rate=0.02)
    riskier = RiskEngine(recovery_rate=0.4, hazard_rate=0.10)

    cva = lambda e: e.compute_xva_risk(ConstantSurrogate(5.0), paths, time_grid, R)[
        "CVA"
    ]

    assert abs(cva(no_recovery) - cva(base) / 0.6) < 1e-10
    assert cva(riskier) > cva(base)


def test_reference_engine_agrees_with_a_perfect_surrogate():
    engine = RiskEngine()
    time_grid = jnp.linspace(0.0, 1.0, N_STEPS)
    paths = _feature_paths(time_grid)

    value_fn = _analytic_value_fn()

    class AnalyticSurrogate:
        def predict_price(self, x):
            return value_fn(x)

    surrogate_result = engine.compute_xva_risk(AnalyticSurrogate(), paths, time_grid, R)
    reference = engine.compute_xva_risk_reference(value_fn, paths, time_grid, R)

    assert abs(surrogate_result["CVA"] - reference["CVA"]) < 1e-10
    assert abs(surrogate_result["NetXVA"] - reference["NetXVA"]) < 1e-10


def test_reference_pricer_is_injected_not_assumed():
    engine = RiskEngine()
    time_grid = jnp.linspace(0.0, 1.0, N_STEPS)
    paths = _feature_paths(time_grid)

    constant = engine.compute_xva_risk_reference(
        lambda x: jnp.asarray(7.0) + 0.0 * jnp.sum(x), paths, time_grid, R
    )

    assert bool(jnp.allclose(constant["V_matrix"], 7.0))
    assert constant["CVA"] > 0.0
    assert constant["DVA"] == 0.0


def _analytic_value_fn():
    """The closed form the reference engine used to hardcode."""

    def value_fn(x):
        return black_scholes_price_single(
            spot=x[0],
            strike=x[1],
            maturity=jnp.maximum(x[2], 1e-8),
            sigma=x[3],
            r=x[4],
            is_call=True,
        )

    return value_fn


def test_risk_report_runs():
    RiskEngine().report({"CVA": 1.0, "DVA": 0.5, "NetXVA": 0.5})


def _full_history():
    return {
        "train_loss": [1.0, 0.5, 0.25, 0.2],
        "valid_loss": [1.1, 0.6, 0.3, 0.25],
        "train_price_rmse": [2.0, 1.0, 0.5, 0.4],
        "valid_price_rmse": [2.1, 1.1, 0.6, 0.5],
        "train_price_loss": [0.9, 0.4, 0.2, 0.15],
        "valid_price_loss": [1.0, 0.5, 0.25, 0.2],
        "train_gradient_loss": [0.5, 0.3, 0.1, 0.05],
        "valid_gradient_loss": [0.6, 0.35, 0.15, 0.1],
        "train_hessian_loss": [0.2, 0.1, 0.05, 0.02],
        "valid_hessian_loss": [0.25, 0.12, 0.06, 0.03],
    }


def _with_temp_logger(body):
    previous = Visualizer.logger
    try:
        with tempfile.TemporaryDirectory() as tmp:
            Visualizer.set_logger(TempLogger(tmp))
            body(tmp)
    finally:
        Visualizer.set_logger(previous)


def test_training_history_plot_is_written():
    def body(tmp):
        Visualizer.plot_training_history(_full_history(), filename="history.png")
        assert os.path.getsize(os.path.join(tmp, "history.png")) > 0

    _with_temp_logger(body)


def test_training_history_plot_handles_unused_sobolev_terms():
    def body(tmp):
        history = _full_history()
        history["train_hessian_loss"] = [0.0, 0.0, 0.0, 0.0]
        history["valid_hessian_loss"] = [0.0, 0.0, 0.0, 0.0]
        history["train_gradient_loss"] = []

        Visualizer.plot_training_history(history, filename="partial.png")

        assert os.path.exists(os.path.join(tmp, "partial.png"))

    _with_temp_logger(body)


def test_mc_path_plot_accepts_2d_and_3d_paths():
    def body(tmp):
        time_grid = jnp.linspace(0.0, 1.0, 6)

        Visualizer.plot_mc_paths(time_grid, jnp.ones((4, 6, 1)), filename="p3.png")
        Visualizer.plot_mc_paths(time_grid, jnp.ones((4, 6)), filename="p2.png")

        assert os.path.exists(os.path.join(tmp, "p3.png"))
        assert os.path.exists(os.path.join(tmp, "p2.png"))

    _with_temp_logger(body)


def test_price_comparison_and_exposure_plots():
    def body(tmp):
        Visualizer.plot_price_comparison(
            jnp.array([1.0, 2.0, 3.0]), jnp.array([1.1, 1.9, 3.2]), filename="pc.png"
        )

        time_grid = jnp.linspace(0.0, 1.0, 5)
        Visualizer.plot_exposure_profiles(
            time_grid, jnp.ones(5), jnp.zeros(5), filename="exp.png"
        )

        assert os.path.exists(os.path.join(tmp, "pc.png"))
        assert os.path.exists(os.path.join(tmp, "exp.png"))

    _with_temp_logger(body)


def test_surrogate_surface_plot():
    def body(tmp):
        network = eqx.nn.MLP(
            5, "scalar", 8, 1, activation=jax.nn.softplus, key=jax.random.PRNGKey(0)
        )

        Visualizer.plot_surrogate_surface(
            surrogate=SurrogateModel(network),
            fixed_input=jnp.array([SPOT, STRIKE, 1.0, SIGMA, R]),
            x_idx=0,
            y_idx=1,
            x_range=(80.0, 120.0),
            y_range=(80.0, 120.0),
            feature_labels=("S1", "S2", "S3", "K", "T"),
            grid_points=6,
            filename="surface.png",
        )

        assert os.path.exists(os.path.join(tmp, "surface.png"))

    _with_temp_logger(body)


def test_surface_labels_come_from_the_caller_not_a_fixed_table():
    labels = ("S1", "S2", "S3", "K", "T")

    assert Visualizer._feature_label(labels, 1) == "S2"
    assert Visualizer._feature_label(labels, 4) == "T"
    assert Visualizer._feature_label(labels, 9) == "Input 9"
    assert Visualizer._feature_label((), 0) == "Input 0"


def test_rolling_average_handles_short_and_oversized_windows():
    short = Visualizer._rolling_average([1.0, 2.0], window=10)

    assert len(short) == 2

    smoothed = Visualizer._rolling_average([1.0, 2.0, 3.0, 4.0], window=2)

    assert len(smoothed) == 3
    assert abs(float(smoothed[0]) - 1.5) < 1e-12


def test_metric_reports_run():
    Visualizer.report_metrics({"RMSE": 1.0, "Note": "text is skipped"})
    Visualizer.report_risk_metrics({"CVA": 1.0, "DVA": 0.5, "NetXVA": 0.5})


if __name__ == "__main__":
    for check in [
        test_positive_exposure_produces_cva_and_no_dva,
        test_negative_exposure_produces_dva_and_no_cva,
        test_exposures_are_non_negative_and_net_is_consistent,
        test_recovery_and_hazard_rates_scale_the_adjustment,
        test_reference_engine_agrees_with_a_perfect_surrogate,
        test_reference_pricer_is_injected_not_assumed,
        test_risk_report_runs,
        test_training_history_plot_is_written,
        test_training_history_plot_handles_unused_sobolev_terms,
        test_mc_path_plot_accepts_2d_and_3d_paths,
        test_price_comparison_and_exposure_plots,
        test_surrogate_surface_plot,
        test_surface_labels_come_from_the_caller_not_a_fixed_table,
        test_rolling_average_handles_short_and_oversized_windows,
        test_metric_reports_run,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
