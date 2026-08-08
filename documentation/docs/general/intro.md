# Introduction

## Motivation

Many applications in quantitative finance rely on mathematical models to price financial derivatives such as options. These models provide highly accurate results but often become computationally expensive when they have to be evaluated repeatedly.

Typical applications such as calibration, optimization, and risk analysis require thousands or even millions of pricing function evaluations. As a result, the computational cost quickly becomes the main bottleneck.

To overcome this limitation, surrogate models can be used. A surrogate model approximates the behavior of a computationally expensive mathematical model while requiring only a fraction of the computation time.

---

## Project Goal

The objective of this project is to develop a neural network surrogate for option pricing using **Higher-Order Sobolev Training**.

Unlike conventional supervised learning, Sobolev Training incorporates not only the function values but also derivative information into the learning process. This additional information allows the neural network to better capture the behavior of the original pricing model.

The resulting surrogate should provide predictions that remain close to the original model while significantly reducing computational cost.

---

## Project Workflow

The project follows the workflow illustrated below.

```text
Pricing Model
      │
      ▼
Generate Training Data
      │
      ▼
Sobolev Training
      │
      ▼
Neural Network Surrogate
      │
      ▼
Validation
      │
      ├──────────────┐
      ▼              ▼
Calibration   Risk Analysis
```

The workflow begins by generating training data from a differentiable option pricing model.

This data is then used to train a neural network surrogate using Higher-Order Sobolev Training.

After training, the surrogate is validated against the original pricing model before being applied to downstream tasks such as calibration and risk analysis.

---

## Repository Structure

The repository is divided into several independent modules.

### Market Simulation

Generates the training and validation data using the selected pricing model.

### Surrogate Modeling

Implements the neural network architecture and the complete training process.

### Calibration

Uses the trained surrogate to estimate model parameters from market observations.

### Risk Visualization

Provides visualization tools for predictions and risk-related analyses.

### Utilities

Contains helper functions that are shared across multiple modules.

The modular organization improves readability, maintainability, and extensibility of the project.

---

## Purpose of this Documentation

This documentation explains both the theoretical background and the software implementation of the project.

Its goal is to help developers, students, and supervisors understand the overall architecture, workflow, and functionality of the software without requiring a detailed inspection of the source code.