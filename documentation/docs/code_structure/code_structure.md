# Code Structure

## Overview

The project is organized into several independent modules, each responsible for a specific part of the overall workflow.

The modular architecture separates market data generation, surrogate model training, calibration, visualization, and shared utility functions. This separation improves readability, maintainability, and extensibility.

---

## Overall Workflow

The software follows the workflow illustrated below.

```text
Market Simulation
        │
        ▼
Training Data
        │
        ▼
Surrogate Modeling
        │
        ▼
Trained Surrogate
        │
        ├──────────────┐
        ▼              ▼
Calibration   Risk Visualization
```

The **Market Simulation** module generates the data required to train the surrogate model.

The **Surrogate Modeling** module learns the behavior of the original pricing model using neural networks and Higher-Order Sobolev Training.

Once training has finished, the surrogate model can be used by different application modules such as **Calibration** and **Risk Visualization**.

---

## Repository Organization

The repository is divided into several modules, each with a clearly defined responsibility.

| Module | Responsibility |
|---------|----------------|
| `pipeline` | Runs the experiment end to end and holds all configuration. |
| `marktsimulation` | Generates market and training data. |
| `surrogate_modeling` | Implements the neural network, the training process and the pricing problems. |
| `kalibrierung` | Performs parameter calibration using market data. |
| `risk_visualisierung` | Visualizes pricing results and risk analyses. |
| `utils` | Provides helper functions shared across multiple modules. |

Each module can be developed independently while interacting through clearly defined interfaces.

The dependency direction is strictly one way, and there are no cycles:

```text
main.py
   └── pipeline
         ├── kalibrierung
         ├── marktsimulation
         ├── surrogate_modeling ──> kalibrierung, marktsimulation
         └── risk_visualisierung ──> surrogate_modeling
```

`pipeline` is the only module that knows the order of the experiment. It never
asks which model is being priced: every stage takes a `PricingProblem` and asks
it. That is what makes a new model a local change - see
*Adding a pricing model*.

---

## Design Principles

The software architecture follows several important software engineering principles.

### Separation of Concerns

Each module is responsible for a single well-defined task.

### Modularity

Independent modules simplify maintenance, testing, and future extensions.

### Reusability

Reusable components reduce duplicated code and simplify future developments.

### Extensibility

New pricing models, neural network architectures, or visualization methods can be added without major changes to the existing codebase.

---

## Module Documentation

The following pages describe the implementation of each module in detail.

- Calibration
- Market Simulation
- Surrogate Modeling
- Pricing Problems
- Risk Visualization
- Utilities
- Pipeline
- Adding a pricing model

Each module documentation explains

- its purpose,
- its internal structure,
- the main classes,
- the most important methods,
- and how it interacts with the rest of the project.

---

## Summary

The modular design of the project separates different responsibilities into dedicated components.

This architecture improves readability, simplifies maintenance, and allows individual modules to evolve independently while remaining part of a coherent software system.