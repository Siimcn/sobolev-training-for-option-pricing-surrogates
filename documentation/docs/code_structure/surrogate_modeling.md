# Surrogate Modeling

## Overview

The **Surrogate Modeling** module learns a cheap approximation of an expensive Monte-Carlo pricing model. Instead of only matching prices, it also matches their derivatives, which is what Higher-Order Sobolev Training means in this project.

The module consists of the following components.

| File | Responsibility |
|------|----------------|
| `architectures.py` | Built-in network architectures and the architecture registry. |
| `neuralnetwork.py` | Adapters that make foreign networks usable in the pipeline. |
| `surrogate_model.py` | Normalization wrapper and the derivative API (price, gradient, Hessian, HVP). |
| `dataset.py` | `SobolevDataset`, batching, train/test split. |
| `data_generation.py` | Monte-Carlo-labelled training set generation. |
| `losses.py` | Price, gradient and HVP losses and their convex combination. |
| `metrics.py` | Evaluation metrics and per-dimension Greek diagnostics. |
| `sobolev_trainer.py` | Training loop, early stopping, checkpointing. |
| `training_config.py` | Configuration container with validation. |

---

## Module Workflow

```text
Monte Carlo pricer
        │
        ▼
SobolevDataset  (prices, gradients, HVPs)
        │
        ▼
SurrogateModel  (normalization + derivative API)
        │
        ▼
SobolevTrainer  (convex price/gradient/HVP loss)
        │
        ▼
Trained surrogate  ->  Calibration, Risk Visualization
```

---

## The Model Contract

The library is **architecture-agnostic, not model-agnostic**.

`SurrogateModel` accepts any callable JAX model. No base class has to be subclassed and no interface has to be implemented. In return, the model has to satisfy six requirements. Everything that satisfies them works, regardless of architecture family; everything that violates one of them fails, regardless of how simple it is.

### Requirements

| # | Requirement | Enforced by |
|---|-------------|-------------|
| 1 | **Callable** as `model(x)` with a single positional argument and no random key | `SurrogateModel.__call__` |
| 2 | **Single-sample input** of shape `(d,)`, not a batch | `predict_prices` applies `jax.vmap` |
| 3 | **Array output**, not a tuple or a dictionary | the denormalization `y * y_std + y_mean` |
| 4 | **Scalar output** after `squeeze()`, so `out_size=1` or `out_size="scalar"` | `jax.grad` |
| 5 | **Reverse-mode differentiable** | `predict_gradient` |
| 6 | **Twice differentiable** when `sobolev_order = 2` | `predict_hessian`, `predict_hvp` |
| 7 | **Trainable parameters as inexact array leaves** of the model PyTree | `SobolevTrainer` optimizer state |

A keyword-only `key` argument with a default, as in `def __call__(self, x, *, key=None)`, satisfies requirement 1. This is the Equinox convention and the signature used by `diff_ml.nn.Normalized`.

Requirement 7 means float arrays. Boolean masks and integer buffers are allowed as model fields, but they are not treated as parameters: the optimizer state is built with `equinox.is_inexact_array`, matching the leaves that `equinox.filter_value_and_grad` actually differentiates.

### Architectures that satisfy the contract

Verified in `tests/test_surrogate_architectures.py`, each without any modification to the library:

- multi-layer perceptrons (`equinox.nn.MLP`),
- residual networks with skip connections (`ResidualMLP`),
- attention and transformer blocks (`equinox.nn.MultiheadAttention`, `LayerNorm`),
- convolutional networks (`equinox.nn.Conv1d`),
- operator networks of the DeepONet branch/trunk form,
- hand-written `equinox.Module` classes,
- plain Python functions,
- networks whose parameters live outside the model (Flax, Haiku), through `FunctionalNetwork`.

### Patterns that do not satisfy the contract

| Pattern | Requirement violated |
|---------|----------------------|
| Stateful layers such as `equinox.nn.BatchNorm`, called as `model(x, state)` | 1 and 3 |
| Networks returning a tuple, for example a twin network returning `(value, derivative)` | 3 |
| Vector-valued output with `out_size > 1` | 4 |
| Models expecting a leading batch axis, the usual Flax convention | 2 |
| `Dropout` and other layers requiring a random key at call time | 1 |
| `jax.lax.while_loop` in the forward pass | 5 |

The first two patterns can be supported with a thin adapter but are deliberately not supported today. A pricing surrogate is expected to be deterministic, which is why layers needing runtime randomness are out of scope; pass such a network in inference mode instead.

---

## Second-Order Sobolev Training and Activation Functions

This is a fundamental limitation rather than an implementation detail.

Second-order Sobolev training differentiates the network twice. A **piecewise-linear activation has a second derivative of zero almost everywhere**, so the network's HVP output is identically zero. The optimizer then trains against a constant it has no way to reduce.

The critical part is that **nothing raises an error**. First-order training keeps working normally, the gradient term still improves, and only the HVP term silently stalls.

| Activation | First order | Second order |
|------------|-------------|--------------|
| `relu`, `leaky_relu`, `hard_tanh` | works | **identically zero** |
| `softplus`, `silu`, `gelu`, `tanh`, `elu` | works | works |

The defaults in `architectures.py` are smooth for this reason. The behaviour is pinned down in `tests/test_second_order_sobolev.py`, which asserts both that piecewise-linear activations produce a zero Hessian and that the HVP loss cannot improve during training.

The same applies to `jax.lax.stop_gradient` in the forward pass, which zeroes the first and second derivatives alike, and to the payoff inside the Monte-Carlo pricer: the smoothing in `marktsimulation/payoff.py` exists so that the HVP **labels** are not zero for the same reason.

---

## architectures.py

Named architectures are looked up in a registry, so a new one can be added from outside without changing library code.

```python
register_architecture("MYNET", MyEquinoxModule)
network = build_network("MYNET", key, in_size=5, width_size=64)
```

A builder is called as `builder(key=..., in_size=..., out_size=..., **kwargs)` and must return a callable JAX model. Names are case-insensitive, and re-registering an existing name requires `overwrite=True` so a custom architecture cannot silently shadow a built-in one.

The registry is a convenience only. `SurrogateModel` takes an already-built network directly, so nothing has to be registered to be usable.

Built-in architectures are `MLP` and `RESMLP`.

---

## neuralnetwork.py

Two optional adapters.

`NeuralNetwork` wraps any callable model and adds a readable `architecture` label. `NeuralNetwork.from_architecture(...)` builds a registered architecture and wraps it in one step.

`FunctionalNetwork` closes the one calling convention a foreign network can trip over. Equinox keeps parameters inside the model, so it is called as `model(x)`. Flax and Haiku keep them apart and are applied as `apply_fn(params, x)`. The adapter holds the parameter PyTree as its own differentiable leaves.

```python
network = FunctionalNetwork(variables, lambda p, x: module.apply(p, x))
```

Anything else the foreign model needs, such as a batch axis, a random key or `deterministic=True`, belongs inside that closure.

---

## surrogate_model.py

`SurrogateModel` normalizes inputs and denormalizes outputs so the network trains on quantities of order one. The raw features `[S, K, T, sigma, r]` and the price target span very different magnitudes, which leaves training ill-conditioned otherwise.

Derivatives are still returned with respect to the **raw** input, because JAX carries the chain rule through the affine normalization automatically.

The statistics default to the identity, which turns normalization off. Use that for a model that already normalizes itself, such as one wrapping its own normalization layers, so it is not normalized twice.

The derivative API is

- `predict_price`, `predict_gradient`, `predict_hessian` for a single sample,
- `predict_prices`, `predict_gradients`, `predict_hessians` for a batch,
- `predict_hvp`, `predict_hvps` for Hessian-vector products.

The HVP path uses forward-over-reverse differentiation and never materializes the full Hessian, which matters because the Hessian is quadratic in the input dimension.

---

## sobolev_trainer.py

`SobolevTrainer` minimizes a convex combination of the price, gradient and HVP losses. The weights are computed in `losses.sobolev_loss_weights` following Savine's *Differential Machine Learning*, extended from price and gradient to price, gradient and HVP.

Gradients and HVPs are rescaled by `x_std / y_std` before entering the loss, so the three residuals share a common scale despite their different natural units. An optional per-dimension rescaling prevents whichever input dimension has the largest spread from dominating the pooled error.

The trainer consumes a `SurrogateModel`, not a bare network: it reads `x_std` and `y_std` from it and calls its derivative API.

`sobolev_order` selects which terms take part: `0` for price only, `1` for price and gradient, `2` for price, gradient and HVP.

---

## Summary

The surrogate pipeline places no restriction on the network architecture. It places a small number of precise restrictions on the model **interface**: one sample in, one scalar out, differentiable as often as the chosen Sobolev order requires, with float parameters in its PyTree.

Everything meeting that contract can be trained, evaluated and used for risk analysis without changing a line of the library.
