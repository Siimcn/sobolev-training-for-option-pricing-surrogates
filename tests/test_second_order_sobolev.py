"""A piecewise-linear activation has a second derivative of zero almost
everywhere, so the HVP term is identically zero and nothing raises.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.sobolev_trainer import SobolevTrainer
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.training_config import TrainingConfig


D = 4
X = jnp.linspace(0.6, 1.4, D)
V = jnp.ones(D) / jnp.sqrt(D)

PIECEWISE_LINEAR = {
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "hard_tanh": jax.nn.hard_tanh,
}

SMOOTH = {
    "softplus": jax.nn.softplus,
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "elu": jax.nn.elu,
}


def _surrogate(activation, seed=0):
    network = eqx.nn.MLP(
        D, "scalar", 32, 3, activation=activation, key=jax.random.PRNGKey(seed)
    )
    return SurrogateModel(network)


def _dataset(n=48, seed=0):
    x_key, v_key = jax.random.split(jax.random.PRNGKey(seed))

    X_data = jax.random.uniform(x_key, (n, D), minval=0.6, maxval=1.4)

    def f(x):
        return jnp.sum(x**2)

    V_raw = jax.random.normal(v_key, (n, D))
    V_data = V_raw / jnp.linalg.norm(V_raw, axis=1, keepdims=True)

    return SobolevDataset(
        X=X_data,
        y=jax.vmap(f)(X_data),
        gradients=jax.vmap(jax.grad(f))(X_data),
        hvps=jax.vmap(
            lambda x, v: jax.jvp(jax.grad(f), (x,), (v,))[1]
        )(X_data, V_data),
        V=V_data,
    )


def _fit(activation, epochs=15):
    config = TrainingConfig(
        learning_rate=1e-2,
        batch_size=16,
        epochs=epochs,
        early_stopping=False,
        print_every=epochs,
        sobolev_order=2,
    )

    trainer = SobolevTrainer(_surrogate(activation), config)

    return trainer.fit(_dataset())


def test_piecewise_linear_activations_give_zero_second_order():
    for name, activation in PIECEWISE_LINEAR.items():
        surrogate = _surrogate(activation)

        hvp = surrogate.predict_hvp(X, V)
        hessian = surrogate.predict_hessian(X)

        assert float(jnp.max(jnp.abs(hvp))) == 0.0, name
        assert float(jnp.max(jnp.abs(hessian))) == 0.0, name


def test_piecewise_linear_first_order_is_unaffected():
    # order 1 keeps working, which is why this fails silently rather than loudly
    for name, activation in PIECEWISE_LINEAR.items():
        gradient = _surrogate(activation).predict_gradient(X)

        assert float(jnp.linalg.norm(gradient)) > 0.0, name


def test_smooth_activations_give_nonzero_hvp():
    for name, activation in SMOOTH.items():
        surrogate = _surrogate(activation)

        hvp = surrogate.predict_hvp(X, V)
        hessian = surrogate.predict_hessian(X)

        assert float(jnp.linalg.norm(hvp)) > 0.0, name
        assert float(jnp.linalg.norm(hessian)) > 0.0, name
        assert bool(jnp.all(jnp.isfinite(hessian))), name


def test_hvp_matches_hessian_product_for_smooth_activations():
    for name, activation in SMOOTH.items():
        surrogate = _surrogate(activation)

        direct = surrogate.predict_hessian(X) @ V
        cheap = surrogate.predict_hvp(X, V)

        assert float(jnp.max(jnp.abs(direct - cheap))) < 1e-9, name


def test_relu_hessian_loss_cannot_improve():
    # predictions are identically zero, so the term is a constant
    history = _fit(jax.nn.relu)

    hessian_losses = history["train_hessian_loss"]

    assert hessian_losses[0] > 0.0
    assert abs(hessian_losses[-1] - hessian_losses[0]) < 1e-12


def test_smooth_hessian_loss_does_improve():
    history = _fit(jax.nn.softplus)

    hessian_losses = history["train_hessian_loss"]

    assert hessian_losses[0] > 0.0
    assert hessian_losses[-1] < hessian_losses[0]


if __name__ == "__main__":
    for check in [
        test_piecewise_linear_activations_give_zero_second_order,
        test_piecewise_linear_first_order_is_unaffected,
        test_smooth_activations_give_nonzero_hvp,
        test_hvp_matches_hessian_product_for_smooth_activations,
        test_relu_hessian_loss_cannot_improve,
        test_smooth_hessian_loss_does_improve,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
