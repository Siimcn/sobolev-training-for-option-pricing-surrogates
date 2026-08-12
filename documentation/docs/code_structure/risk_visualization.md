# Risk Visualization

## Overview

The **Risk Visualization** module is responsible for evaluating and presenting the financial risk associated with option pricing models.

After the surrogate model has been trained and calibrated, this module estimates risk measures such as **Expected Positive Exposure (EPE)**, **Expected Negative Exposure (ENE)**, **Credit Valuation Adjustment (CVA)**, **Debit Valuation Adjustment (DVA)**, and **Net XVA**.

Besides performing the risk calculations, the module also provides visualization utilities that simplify the interpretation of simulation results and model performance.

The module consists of three main components:

- `riskengine.py` – computes exposure profiles and XVA metrics.
- `visualizer.py` – generates plots and reports.
- `xva_analysis.py` – executes the complete XVA analysis workflow.

Together, these components transform simulated market scenarios into meaningful risk metrics and graphical representations.

---

## Module Workflow

The overall workflow of the module is illustrated below.

```text
Monte Carlo Simulation
          │
          ▼
Generated Asset Paths
          │
          ▼
Feature Construction
          │
          ▼
Surrogate Evaluation
          │
          ▼
RiskEngine
          │
          ▼
Exposure Profiles
(EPE / ENE)
          │
          ▼
CVA / DVA / Net XVA
          │
          ▼
Visualizer
          │
          ▼
Plots and Reports
```

The module first evaluates the trained surrogate model on Monte Carlo simulation paths. Based on these predictions, exposure profiles are computed and transformed into XVA risk measures. Finally, the results are visualized and summarized.

---

## riskengine.py

### Purpose

The `RiskEngine` class is the computational core of the module.

Its responsibility is to evaluate simulated option values and calculate exposure profiles together with the corresponding XVA metrics.

The implementation supports both surrogate-based pricing and reference pricing using the analytical Black-Scholes model, allowing direct validation of the surrogate model.

---

### Main Responsibilities

The `RiskEngine` is responsible for

- evaluating option values along simulated market paths,
- computing discounted portfolio values,
- calculating Expected Positive Exposure (EPE),
- calculating Expected Negative Exposure (ENE),
- estimating default probabilities,
- computing CVA, DVA and Net XVA,
- comparing surrogate-based and analytical pricing results.

---

### Risk Calculation

The XVA calculation follows several consecutive steps.

1. Evaluate the option value for every simulated market state.
2. Discount all future values to the present.
3. Compute positive and negative exposure profiles.
4. Estimate default probabilities using the hazard rate.
5. Calculate the loss given default (LGD).
6. Compute the final CVA, DVA and Net XVA values.

This modular workflow allows different pricing models to be integrated without modifying the risk calculation itself.

---

### Reporting

The `report()` method prints a concise summary of the computed risk measures, including

- Credit Valuation Adjustment (CVA),
- Debit Valuation Adjustment (DVA),
- Net XVA.

This provides a quick overview of the calculated counterparty risk.

---

## visualizer.py

### Purpose

The `Visualizer` class provides a collection of plotting and reporting utilities used throughout the project.

Instead of performing numerical computations, it focuses on presenting results in an intuitive and easily interpretable way.

All generated figures can either be stored automatically or displayed directly during program execution.

---

### Main Responsibilities

The visualizer is responsible for generating figures and reports for different stages of the project.

Its functionality includes

- plotting training history,
- visualizing Monte Carlo paths,
- comparing surrogate predictions with reference prices,
- displaying exposure profiles,
- generating three-dimensional surrogate price surfaces,
- printing evaluation and risk reports.

---

### Generated Visualizations

The module supports several types of visualizations.

| Visualization | Purpose |
|---------------|---------|
| Training History | Displays loss and RMSE during neural network training. |
| Monte Carlo Paths | Shows simulated asset price trajectories. |
| Price Comparison | Compares surrogate predictions with reference prices. |
| Exposure Profiles | Displays Expected Positive and Negative Exposure over time. |
| Surrogate Surface | Visualizes the learned pricing function in three dimensions. |

These visualizations simplify both debugging and performance evaluation of the surrogate model.

---

## xva_analysis.py

### Purpose

The `xva_analysis.py` module coordinates the complete XVA evaluation workflow.

Rather than implementing new pricing algorithms, it combines the different project components into a single analysis pipeline.

---

### Workflow

The workflow consists of the following steps.

1. Generate Monte Carlo asset paths.
2. Construct surrogate input features.
3. Evaluate the surrogate model.
4. Compute XVA metrics using the `RiskEngine`.
5. Compute analytical reference values.
6. Compare surrogate and reference results.
7. Visualize exposure profiles and generate reports.

This workflow provides a direct validation of the surrogate model by comparing its risk estimates with those obtained from the analytical pricing model.

---

### Validation

To assess the quality of the surrogate model, the implementation compares

- CVA,
- DVA,
- Net XVA

between surrogate predictions and analytical Black-Scholes pricing.

Relative errors are computed and reported, allowing the accuracy of the surrogate model to be evaluated quantitatively.

---

## Design Considerations

The module follows a clear separation of responsibilities.

- `RiskEngine` performs all numerical risk calculations.
- `Visualizer` focuses exclusively on presenting results.
- `xva_analysis` coordinates the complete analysis pipeline.

This modular architecture improves maintainability and allows individual components to be extended independently.

---

## Summary

The Risk Visualization module combines Monte Carlo simulation, surrogate model evaluation, and XVA analysis into a unified workflow.

By separating numerical computations from visualization and workflow management, the module provides a flexible framework for evaluating counterparty risk while also offering intuitive graphical representations of the obtained results.