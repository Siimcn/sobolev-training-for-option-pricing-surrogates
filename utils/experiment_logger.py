import json
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime

from utils.paths import project_path


def _json_safe(value):
    """Coerce a possibly nested config into JSON types."""

    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_safe(getattr(value, f.name)) for f in fields(value)}

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (bool, int, float, str)) or value is None:
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)

    return str(value)


class LoggerWriter:
    """Tee für sys.stdout: schreibt in Konsole und Logdatei."""

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

    def __init__(self, base_dir: str = "results", echo: bool = True):

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # anchored to the repository, not to the working directory, so a run
        # started from anywhere writes to the same place
        self.output_dir = project_path(base_dir, self.run_id)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        sys.stdout = LoggerWriter(self.path("console_output.txt"), echo=echo)

    def path(self, filename: str):

        return self.output_dir / filename

    def save_calibration(self, fitted_params, spot, filename: str = "calibration.json"):

        fields = (
            fitted_params._asdict()
            if hasattr(fitted_params, "_asdict")
            else {"params": fitted_params}
        )

        data = {"Spot": float(spot), **_json_safe(dict(fields))}

        with open(self.path(filename), "w") as f:

            json.dump(data, f, indent=4)

    def save_metrics(self, metrics: dict, filename: str = "metrics.json"):

        serializable = {}

        for k, v in metrics.items():

            try:
                serializable[k] = float(v)
            except Exception:
                serializable[k] = str(v)

        with open(self.path(filename), "w") as f:

            json.dump(serializable, f, indent=4)

    def save_xva(self, xva: dict, filename: str = "xva.json"):

        serializable = {}

        for k, v in xva.items():

            try:
                serializable[k] = float(v)

            except Exception:
                pass

        with open(self.path(filename), "w") as f:

            json.dump(serializable, f, indent=4)

    def save_config(self, config, filename: str = "config.json"):

        with open(self.path(filename), "w") as f:
            json.dump(_json_safe(config), f, indent=4)

    def save_report(self, text: str, filename: str = "report.txt"):

        with open(self.path(filename), "w") as f:

            f.write(text)

    def print_location(self):

        print(f"\nResults stored in:\n" f"{self.output_dir}\n")
