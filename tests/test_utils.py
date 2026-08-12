import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from kalibrierung.market_data import MarketData
from marktsimulation.pricing_model import BlackScholesParams
from pipeline.config import BasketConfig, DataConfig, ExperimentConfig
from surrogate_modeling.pricing_problem import CalibrationResult, build_problem
from surrogate_modeling.training_config import TrainingConfig
from utils.paths import PROJECT_ROOT, project_path
from utils.experiment_logger import ExperimentLogger, LoggerWriter


def _problem(config):
    market_data = MarketData(
        spot=100.0,
        strikes=jnp.array([80.0, 100.0, 120.0]),
        maturities=jnp.array([0.25, 0.5, 1.0]),
        market_prices=jnp.ones(3),
        is_call=jnp.array([True, True, True]),
    )

    return build_problem(
        config.data.pricing_model,
        config=config,
        market_data=market_data,
        calibration=CalibrationResult(
            params=BlackScholesParams(r=0.05, sigma=0.2),
            assumptions={"correlation": 0.5},
        ),
    )


@contextmanager
def _logger(echo=True):
    """
    ExperimentLogger replaces the process-wide sys.stdout in its constructor
    and never closes the file it opens, so a test has to undo both or Windows
    cannot remove the temporary directory.
    """

    original_stdout = sys.stdout

    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(base_dir=tmp, echo=echo)
        try:
            yield logger, tmp
        finally:
            if isinstance(sys.stdout, LoggerWriter):
                sys.stdout.log_file.close()
            sys.stdout = original_stdout


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_run_directory_is_created_under_base_dir():
    with _logger() as (logger, tmp):
        assert os.path.isdir(logger.output_dir)
        assert os.path.dirname(logger.output_dir) == tmp
        assert len(logger.run_id) == len("20240101_120000")


def test_path_joins_into_the_run_directory():
    """
    Compared as paths, not as strings: `path()` returns a `pathlib.Path` so
    that callers work regardless of the platform separator.
    """

    with _logger() as (logger, _):
        assert Path(logger.path("a.png")) == Path(logger.output_dir) / "a.png"


def test_output_directory_does_not_depend_on_the_working_directory(tmp_path):
    """
    A run launched from anywhere writes to the same place. `results/` used to
    be resolved against `os.getcwd()`, so starting the pipeline from another
    directory scattered the artifacts.
    """

    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        assert project_path("results") == PROJECT_ROOT / "results"
        assert project_path("results").is_absolute()
    finally:
        os.chdir(cwd)


def test_project_path_lets_an_absolute_base_win(tmp_path):
    """A caller that already knows where it wants to write is not overridden."""

    assert project_path(str(tmp_path), "run") == tmp_path / "run"


def test_stdout_is_teed_into_the_console_log():
    with _logger() as (logger, _):
        print("marker-line")
        sys.stdout.flush()

        console = logger.path("console_output.txt")

        assert os.path.exists(console)

        with open(console, encoding="utf-8") as handle:
            assert "marker-line" in handle.read()


def test_echo_false_keeps_the_log_but_silences_the_terminal():
    class Recorder:
        def __init__(self):
            self.written = []

        def write(self, text):
            self.written.append(text)

        def flush(self):
            pass

    original_stdout = sys.stdout
    recorder = Recorder()

    try:
        sys.stdout = recorder

        with _logger(echo=False) as (logger, _):
            print("silenced-line")
            sys.stdout.flush()

            console = logger.path("console_output.txt")

            with open(console, encoding="utf-8") as handle:
                logged = handle.read()
    finally:
        sys.stdout = original_stdout

    assert "silenced-line" in logged, "the run log must still be complete"
    assert recorder.written == [], "nothing may reach the terminal"


def test_logger_writer_forwards_to_the_original_stream():
    class Recorder:
        def __init__(self):
            self.written = []

        def write(self, text):
            self.written.append(text)

        def flush(self):
            pass

    original_stdout = sys.stdout
    recorder = Recorder()

    with tempfile.TemporaryDirectory() as tmp:
        try:
            sys.stdout = recorder
            writer = LoggerWriter(os.path.join(tmp, "log.txt"))
            writer.write("hello")
            writer.flush()
        finally:
            writer.log_file.close()
            sys.stdout = original_stdout

    assert recorder.written == ["hello"]


def test_save_metrics_coerces_values_to_json():
    with _logger() as (logger, _):
        logger.save_metrics(
            {"RMSE": jnp.array(1.5), "Count": 3, "Label": "not-a-number"}
        )

        saved = _read_json(logger.path("metrics.json"))

        assert abs(saved["RMSE"] - 1.5) < 1e-12
        assert saved["Count"] == 3.0
        assert saved["Label"] == "not-a-number"


def test_save_xva_drops_non_numeric_entries():
    with _logger() as (logger, _):
        logger.save_xva({"CVA": 1.0, "DVA": 0.25, "NetXVA": 0.75, "EPE": jnp.ones(4)})

        saved = _read_json(logger.path("xva.json"))

        assert set(saved) == {"CVA", "DVA", "NetXVA"}
        assert abs(saved["NetXVA"] - 0.75) < 1e-12


def test_save_config_serializes_the_training_config():
    with _logger() as (logger, _):
        config = TrainingConfig(learning_rate=1e-3, epochs=7, sobolev_order=1)

        logger.save_config(config)

        saved = _read_json(logger.path("config.json"))

        assert abs(saved["learning_rate"] - 1e-3) < 1e-12
        assert saved["epochs"] == 7
        assert saved["sobolev_order"] == 1
        assert saved["early_stopping"] is True


def test_save_config_archives_a_nested_experiment_config():
    config = ExperimentConfig(
        data=DataConfig(pricing_model="basket_black_scholes", min_maturity=0.05),
        basket=BasketConfig(n_assets=3, correlation=0.5, symmetrize=True),
    )

    with _logger() as (logger, _):
        logger.save_config(config.to_dict(_problem(config)))

        saved = _read_json(logger.path("config.json"))

    assert saved["data"]["pricing_model"] == "basket_black_scholes"
    assert abs(saved["data"]["min_maturity"] - 0.05) < 1e-12
    assert saved["data"]["n_samples"] == config.data.n_samples

    assert saved["basket"]["n_assets"] == 3
    assert abs(saved["basket"]["correlation"] - 0.5) < 1e-12
    assert saved["basket"]["symmetrize"] is True
    assert saved["basket"]["weights"] is None

    assert saved["payoff"]["name"] == "european_call"
    assert saved["simulation"]["shared_label_keys"] is False
    assert saved["simulation"]["antithetic"] is True

    assert saved["market"]["ticker"] == "AAPL"
    assert saved["network"]["in_size"] is None
    assert saved["training"]["selection_metric"] == "total"

    assert saved["derived"]["problem"] == "basket_black_scholes"
    assert saved["derived"]["feature_names"] == ["S1", "S2", "S3", "K", "T"]
    assert saved["derived"]["n_features"] == 5

    assert len(saved["derived"]["domain_low"]) == 5
    assert saved["derived"]["domain_low"][4] >= 0.05
    assert saved["derived"]["exchangeable_features"] == [0, 1, 2]

    assert saved["derived"]["assumptions"]["correlation"] == 0.5


def test_two_pricing_models_no_longer_share_a_config_file():
    bs = ExperimentConfig(data=DataConfig(pricing_model="black_scholes"))
    basket = ExperimentConfig(data=DataConfig(pricing_model="basket_black_scholes"))

    bs_dict = bs.to_dict(_problem(bs))
    basket_dict = basket.to_dict(_problem(basket))

    assert bs_dict != basket_dict
    assert (
        bs_dict["derived"]["feature_names"] != basket_dict["derived"]["feature_names"]
    )


def test_config_without_a_problem_omits_the_derived_block():
    assert "derived" not in ExperimentConfig().to_dict()


def test_save_calibration_and_report():
    from marktsimulation.pricing_model import BlackScholesParams, HestonParams

    with _logger() as (logger, _):
        logger.save_calibration(BlackScholesParams(r=0.03, sigma=0.25), spot=101.5)
        logger.save_report("report body")

        saved = _read_json(logger.path("calibration.json"))

        assert abs(saved["Spot"] - 101.5) < 1e-12
        assert abs(saved["sigma"] - 0.25) < 1e-12
        assert abs(saved["r"] - 0.03) < 1e-12

        with open(logger.path("report.txt"), encoding="utf-8") as handle:
            assert handle.read() == "report body"

    with _logger() as (logger, _):
        logger.save_calibration(
            HestonParams(r=0.03, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, nu0=0.04),
            spot=101.5,
        )

        saved = _read_json(logger.path("calibration.json"))

        assert set(saved) == {"Spot", "r", "kappa", "theta", "xi", "rho", "nu0"}
        assert abs(saved["kappa"] - 2.0) < 1e-12


def test_print_location_runs():
    with _logger() as (logger, _):
        logger.print_location()


if __name__ == "__main__":
    for check in [
        test_run_directory_is_created_under_base_dir,
        test_path_joins_into_the_run_directory,
        test_stdout_is_teed_into_the_console_log,
        test_echo_false_keeps_the_log_but_silences_the_terminal,
        test_logger_writer_forwards_to_the_original_stream,
        test_save_metrics_coerces_values_to_json,
        test_save_xva_drops_non_numeric_entries,
        test_save_config_serializes_the_training_config,
        test_save_config_archives_a_nested_experiment_config,
        test_two_pricing_models_no_longer_share_a_config_file,
        test_config_without_a_problem_omits_the_derived_block,
        test_save_calibration_and_report,
        test_print_location_runs,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
