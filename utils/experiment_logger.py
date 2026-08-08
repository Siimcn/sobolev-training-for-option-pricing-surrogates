import os
import json
import sys
from datetime import datetime


class LoggerWriter:
    """
    Tee für sys.stdout: schreibt in Konsole und Logdatei.

    echo=False drops the terminal half, which is how a run is silenced
    without touching any print() call site.
    """

    def __init__(self, filename, echo=True):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", encoding="utf-8")
        self.echo = echo

    def write(self, message):
        if self.echo:
            self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        if self.echo:
            self.terminal.flush()
        self.log_file.flush()


class ExperimentLogger:

    def __init__(
        self,
        base_dir: str = "results",
        echo: bool = True,
    ):

        self.run_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.output_dir = os.path.join(
            base_dir,
            self.run_id,
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        # replaces the process-wide sys.stdout, so every later print() in
        # the run is captured without touching the call sites
        sys.stdout = LoggerWriter(
            self.path("console_output.txt"),
            echo=echo,
        )

    def path(
        self,
        filename: str,
    ):

        return os.path.join(
            self.output_dir,
            filename,
        )
    
    def save_calibration(
        self,
        fitted_params,
        spot,
        filename: str = "calibration.json",
    ):

        data = {
            "Spot": float(spot),
            "Sigma": float(
                fitted_params.sigma
            ),
            "Rate": float(
                fitted_params.r
            ),
        }

        with open(
            self.path(filename),
            "w",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
            )



    def save_metrics(
        self,
        metrics: dict,
        filename: str = "metrics.json",
    ):

        serializable = {}

        for k, v in metrics.items():

            try:
                serializable[k] = float(v)
            except Exception:
                serializable[k] = str(v)

        with open(
            self.path(filename),
            "w",
        ) as f:

            json.dump(
                serializable,
                f,
                indent=4,
            )


    def save_xva(
        self,
        xva: dict,
        filename: str = "xva.json",
    ):

        serializable = {}

        for k, v in xva.items():

            try:
                serializable[k] = float(v)

            except Exception:
                pass

        with open(
            self.path(filename),
            "w",
        ) as f:

            json.dump(
                serializable,
                f,
                indent=4,
            )

    def save_config(
        self,
        config,
        filename: str = "config.json",
    ):

        config_data = {}
        for k, v in vars(config).items():
            try:
                # JAX arrays are not JSON-serializable
                config_data[k] = v.item() if hasattr(v, "item") else v
            except Exception:
                config_data[k] = str(v)

        with open(
            self.path(filename),
            "w",
        ) as f:
            json.dump(
                config_data,
                f,
                indent=4,
            )

    def save_report(
        self,
        text: str,
        filename: str = "report.txt",
    ):

        with open(
            self.path(filename),
            "w",
        ) as f:

            f.write(text)

    def print_location(self):

        print(
            f"\nResults stored in:\n"
            f"{self.output_dir}\n"
        )