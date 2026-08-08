import json
import os
import sys
import tempfile
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp

from surrogate_modeling.training_config import TrainingConfig
from utils.experiment_logger import ExperimentLogger, LoggerWriter


@contextmanager
def _logger(echo=True):
    """ExperimentLogger replaces the process-wide sys.stdout in its
    constructor and never closes the file it opens, so a test has to undo
    both or Windows cannot remove the temporary directory."""

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
    with _logger() as (logger, _):
        assert logger.path("a.png") == os.path.join(logger.output_dir, "a.png")


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
        logger.save_xva(
            {"CVA": 1.0, "DVA": 0.25, "NetXVA": 0.75, "EPE": jnp.ones(4)}
        )

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


def test_save_calibration_and_report():
    class Params:
        sigma = 0.25
        r = 0.03

    with _logger() as (logger, _):
        logger.save_calibration(Params(), spot=101.5)
        logger.save_report("report body")

        saved = _read_json(logger.path("calibration.json"))

        assert abs(saved["Spot"] - 101.5) < 1e-12
        assert abs(saved["Sigma"] - 0.25) < 1e-12
        assert abs(saved["Rate"] - 0.03) < 1e-12

        with open(logger.path("report.txt"), encoding="utf-8") as handle:
            assert handle.read() == "report body"


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
        test_save_calibration_and_report,
        test_print_location_runs,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
