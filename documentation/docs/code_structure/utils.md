# Utilities

## Overview

The **Utilities** module provides supporting functionality that is shared across multiple parts of the project.

Rather than implementing pricing models or machine learning algorithms, this module contains helper classes that simplify experiment management, improve reproducibility, and organize generated results.

Currently, the module consists of the following component:

- `experiment_logger.py` – manages experiment output and stores generated results.

---

## Module Workflow

The utility module supports the execution of the project by automatically organizing experiment outputs.

```text
Project Execution
        │
        ▼
ExperimentLogger
        │
        ├──────────────┐
        ▼              ▼
Create Output     Save Results
 Directory              │
                         ▼
        Calibration • Metrics • XVA
                 • Configuration
                 • Reports
```

Each experiment is stored inside its own timestamped directory, ensuring that multiple runs remain separated and reproducible.

---

## experiment_logger.py

### Purpose

The `ExperimentLogger` class is responsible for managing experiment results generated throughout the project.

Instead of manually creating directories and saving output files, the logger automatically creates a unique experiment folder and provides a consistent interface for storing all generated results.

This simplifies experiment management and makes it easier to compare different runs.

---

### Main Responsibilities

The `ExperimentLogger` is responsible for

- creating a unique output directory,
- organizing experiment files,
- storing calibration results,
- saving evaluation metrics,
- exporting XVA results,
- saving configuration files,
- generating text reports.

---

### Output Organization

Whenever a new experiment starts, the logger automatically creates a timestamp-based directory.

All generated files belonging to the same experiment are stored inside this directory, including

- calibration results,
- performance metrics,
- XVA calculations,
- configuration files,
- textual reports.

This structure keeps project outputs organized and prevents previous experiment results from being overwritten.

---

### Supported Output Files

The logger supports exporting several types of information.

| File | Description |
|------|-------------|
| `calibration.json` | Stores calibrated model parameters. |
| `metrics.json` | Stores evaluation metrics. |
| `xva.json` | Stores calculated XVA risk measures. |
| `config.json` | Stores the experiment configuration. |
| `report.txt` | Stores textual summaries or reports. |

All structured data is exported in JSON format, allowing the results to be reused easily in future analyses.

---

## Design Considerations

The utility module separates experiment management from the numerical algorithms used throughout the project.

By centralizing all file handling inside a dedicated logger, the remaining modules remain focused on their own responsibilities while benefiting from a consistent and reusable storage mechanism.

This design also improves reproducibility by ensuring that every experiment is archived independently.

---

## Summary

The Utilities module provides supporting infrastructure for the project by organizing experiment outputs and storing important results.

Its lightweight design simplifies experiment management while improving the reproducibility, maintainability, and usability of the overall software system.