# Preparation

This section describes how to prepare the project for development and execution.

---

## Prerequisites

Before running the project, the following software should be installed.

- Python 3.11 or newer
- Git
- Visual Studio Code (recommended)
- MkDocs
- MkDocs Material

It is also recommended to use a Python virtual environment.

---

## Clone the Repository

Clone the repository and open it in your preferred development environment.

```bash
git clone <repository-url>
cd <repository-folder>
```

---

## Create a Virtual Environment

To isolate the project dependencies, create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

Install all required Python packages.

```bash
pip install -r requirements.txt
```

If additional packages are required, they can be installed individually.

```bash
pip install <package-name>
```

---

## Project Structure

After installation, the repository consists of several independent modules.

| Module | Description |
|---------|-------------|
| `calibration` | Calibration of option pricing models |
| `market_simulation` | Generation of market data |
| `surrogate_modeling` | Neural network implementation and training |
| `risk_visualization` | Visualization of pricing and risk results |
| `utils` | Shared helper functions |

A detailed explanation of each module is provided in the **Code Structure** section.

---

## Building the Documentation

The documentation is generated using **MkDocs**.

Start the local documentation server with

```bash
mkdocs serve
```

The documentation can then be accessed at

```text
http://127.0.0.1:8000
```

Whenever a Markdown file is saved, MkDocs automatically reloads the page.

---


## Summary

Following these preparation steps ensures that every developer works in a consistent environment and can execute both the project and the documentation without additional configuration.