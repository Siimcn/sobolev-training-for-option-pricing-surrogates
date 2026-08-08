# Calibration

## Overview

The **Calibration** module is responsible for estimating the parameters of an option pricing model such that the model prices match observed market prices as closely as possible.

Instead of manually adjusting model parameters, the calibration process is formulated as a nonlinear least-squares optimization problem. The objective is to minimize the difference between market prices and the prices produced by the pricing model.

The module consists of three main components:

- `calibrator.py` – performs the calibration procedure.
- `market_data.py` – stores and validates market data.
- `market_data_loader.py` – downloads and prepares market data from external sources.

Together, these components provide a complete workflow from obtaining market data to estimating optimized model parameters.

---

## Module Workflow

The calibration process follows the workflow illustrated below.

```text
Market Data
      │
      ▼
MarketDataLoader
      │
      ▼
MarketData
      │
      ▼
Calibrator
      │
      ▼
Pricing Function
      │
      ▼
Residual Computation
      │
      ▼
Levenberg-Marquardt Optimizer
      │
      ▼
Optimized Parameters
      │
      ▼
Error Evaluation
```

The market data is first downloaded and filtered. It is then stored inside a `MarketData` object, which is passed to the `Calibrator`. During optimization, the pricing model is evaluated repeatedly until the difference between model prices and market prices is minimized.

---

## calibrator.py

### Purpose

The `Calibrator` class is the core component of the calibration module.

Its purpose is to estimate the parameters of an option pricing model such that the model reproduces observed market prices as accurately as possible.

The implementation is independent of a specific pricing model. Instead, the pricing model is supplied as a function during initialization, allowing the calibrator to be reused with different pricing models.

---

### Main Responsibilities

The `Calibrator` class is responsible for

- evaluating the pricing model,
- computing weighted residuals,
- performing nonlinear least-squares optimization,
- returning the optimized parameters,
- evaluating the calibration quality.

---

### Constructor

```python
Calibrator(
    pricing_fn,
    transform_fn=None,
    inv_transform_fn=None
)
```

The constructor receives a pricing function that computes option prices for a given parameter set.

Optionally, transformation functions can be provided. These functions map optimization variables to valid model parameters and back again. This is useful when parameters must satisfy constraints such as positivity.

---

### Residual Computation

The private method `_residuals()` computes the weighted difference between

- model prices and
- observed market prices.

If a parameter transformation is defined, the optimization parameters are transformed before the pricing function is evaluated.

The residual vector is multiplied by user-defined weights before it is returned to the optimizer.

---

### Calibration Procedure

The `calibrate()` method performs the complete optimization.

The implementation uses the **Levenberg-Marquardt** algorithm provided by the Optimistix library.

The procedure consists of the following steps:

1. Transform the initial parameter values if necessary.
2. Create the optimization solver.
3. Compute residuals during each optimization step.
4. Minimize the residuals.
5. Transform the optimized parameters back if required.
6. Return both the optimized parameters and the optimization result.

---

### Error Evaluation

The method `pricing_error()` evaluates the quality of the calibrated parameters.

Two error metrics are computed.

| Metric | Description |
|---------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

These metrics measure how accurately the calibrated pricing model reproduces the observed market prices.

---

### Calibration Report

The method `report()` prints a compact summary of the calibration results.

The report currently includes

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

This provides a quick overview of the calibration quality after the optimization has finished.

---

## market_data.py

### Purpose

The `MarketData` class serves as the central data container used throughout the calibration process.

Instead of passing multiple arrays individually, all market-related information is stored inside a single object.

The class is implemented as an `equinox.Module`, making it fully compatible with JAX and Equinox.

---

### Stored Data

A `MarketData` object stores the following information.

| Attribute | Description |
|----------|-------------|
| `spot` | Current price of the underlying asset. |
| `strikes` | Strike prices of the options. |
| `maturities` | Time to maturity of each option. |
| `market_prices` | Observed market prices. |
| `is_call` | Indicates whether each option is a call or a put. |
| `weights` | Optional calibration weights. |

All numerical values are converted to JAX arrays to ensure compatibility with the remaining components of the project.

---

### Data Validation

Before the object is used, all input data is validated.

The validation routine checks that all arrays have identical lengths.

If inconsistent input is detected, a `ValueError` is raised immediately. This prevents invalid datasets from entering the optimization process.

---

### Convenience Functions

The class provides several helper functions that simplify working with market data.

#### Number of Instruments

The property `num_instruments` returns the total number of option contracts stored in the dataset.

#### Call and Put Selection

The properties `calls` and `puts` automatically create new `MarketData` objects containing only call options or only put options.

This allows different subsets of the market data to be processed independently while preserving the original dataset.

#### Summary

The `summary()` method prints a compact overview containing

- spot price,
- number of instruments,
- number of calls,
- number of puts,
- strike range,
- maturity range,
- price range.

This provides a quick consistency check before calibration.

#### Calibration Tuple

The method `calibration_tuple()` returns the stored market data as a tuple.

This representation can easily be passed to other numerical routines if required.

---

## market_data_loader.py

### Purpose

The `MarketDataLoader` class is responsible for obtaining and preparing option market data.

Instead of manually providing option prices, the loader retrieves current market data directly from Yahoo Finance and converts it into a format suitable for the calibration process.

---

### Data Source

The implementation uses the **yfinance** library to access publicly available financial market data.

For a given ticker symbol, the loader retrieves

- the current spot price,
- available option expiration dates,
- option chains,
- bid prices,
- ask prices,
- trading volume,
- open interest.

---

### Data Filtering

Not every available option contract is suitable for calibration.

To improve data quality, several filters are applied.

An option is only accepted if

- the bid price is positive,
- the ask price is positive,
- the ask price is not smaller than the bid price,
- the trading volume exceeds a minimum threshold,
- the open interest exceeds a minimum threshold.

These conditions remove illiquid contracts that could negatively affect the calibration process.

---

### Mid Price Calculation

The implementation uses the midpoint between the bid and ask prices as the market price.

\[
\text{Mid Price}=\frac{\text{Bid}+\text{Ask}}{2}
\]

The midpoint is commonly used because it provides a more robust estimate of the fair market value than either the bid or ask price alone.

---

### Returned Data

After filtering, the loader returns

- spot price,
- strike prices,
- maturities,
- option prices,
- option types.

All numerical values are converted into JAX arrays, allowing them to be used directly by the `MarketData` class and the calibration algorithm.

---

## Design Considerations

The calibration module follows a modular design.

Each class has a clearly defined responsibility:

- `MarketDataLoader` retrieves and prepares market data.
- `MarketData` stores and validates the data.
- `Calibrator` performs the optimization and evaluates the results.

Another important design decision is that the pricing model is **not implemented inside the `Calibrator`**. Instead, it is passed as an external function. This makes the calibration algorithm independent of a specific pricing model and allows the same implementation to be reused with different models.

---

## Summary

The calibration module provides a complete workflow for parameter estimation in option pricing models.

Starting from raw market data, it prepares the dataset, validates the inputs, performs nonlinear least-squares optimization using the Levenberg-Marquardt algorithm, and evaluates the quality of the calibrated parameters.

Its modular architecture improves readability, maintainability, and flexibility, making it easy to extend the project with additional pricing models or calibration strategies in the future.