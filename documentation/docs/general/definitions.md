# Definitions and Terminology

This section introduces the most important concepts used throughout the project.

---

## Option

An **option** is a financial derivative that gives its holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price before or at a specified expiration date.

The objective of option pricing models is to estimate the fair value of these financial contracts.

---

## Option Pricing Model

An **option pricing model** is a mathematical model used to determine the theoretical value of an option.

Typical input parameters include

- underlying asset price,
- strike price,
- time to maturity,
- volatility,
- interest rate.

Examples include the Black-Scholes model and the Heston model.

---

## Surrogate Model

A **surrogate model** is a computational model that approximates the behavior of a more complex mathematical model.

Instead of repeatedly evaluating the original pricing model, the surrogate learns the relationship between inputs and outputs and produces predictions significantly faster.

---

## Neural Network

A **neural network** is a machine learning model consisting of interconnected layers of artificial neurons.

During training, the network learns to approximate the relationship between market parameters and option prices.

Once trained, it can predict option prices for previously unseen input data.

---

## Sobolev Training

**Sobolev Training** extends conventional supervised learning by incorporating derivative information into the training process.

Rather than learning only function values, the neural network is also trained to match the derivatives of the target function.

This additional information often improves approximation accuracy and generalization.

---

## Higher-Order Sobolev Training

**Higher-Order Sobolev Training** further extends this concept by incorporating higher-order derivatives into the loss function.

This enables the surrogate model to capture more detailed characteristics of the original pricing model and can improve both stability and prediction quality.

---

## Calibration

**Calibration** is the process of adjusting the parameters of a mathematical model such that its predictions match observed market data as closely as possible.

In this project, calibration is performed using nonlinear least-squares optimization.

---

## Risk Analysis

**Risk Analysis** investigates how financial models behave under different market conditions.

Because surrogate models are computationally efficient, they enable large-scale simulations that would otherwise be too expensive.

---

## JAX

**JAX** is a Python library for high-performance numerical computing.

It provides

- automatic differentiation,
- vectorized computations,
- just-in-time (JIT) compilation,
- hardware acceleration on CPUs, GPUs, and TPUs.

These capabilities make JAX particularly well suited for implementing Higher-Order Sobolev Training.

---

## Summary

The concepts introduced in this section provide the theoretical foundation for the remaining parts of the documentation.