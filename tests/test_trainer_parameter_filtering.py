import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.sobolev_trainer import SobolevTrainer
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.training_config import TrainingConfig


D = 3


class MaskedMLP(eqx.Module):
    """
    Carries a non-differentiable leaf next to its weights, as a pruned network
    does. `dtype` selects bool (pruning mask) or int (index buffer).
    """

    mlp: eqx.nn.MLP
    mask: jnp.ndarray

    def __init__(self, dtype, *, key):
        mask_key, net_key = jax.random.split(key)

        self.mlp = eqx.nn.MLP(
            D, "scalar", 16, 2, activation=jax.nn.softplus, key=net_key
        )
        self.mask = jax.random.bernoulli(mask_key, 0.8, (D,)).astype(dtype)

    def __call__(self, x):
        return self.mlp(x * self.mask)


class FloatOnlyMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, *, key):
        self.mlp = eqx.nn.MLP(
            D, "scalar", 16, 2, activation=jax.nn.softplus, key=key
        )

    def __call__(self, x):
        return self.mlp(x)


def _dataset(n=32, seed=0):
    x_key, v_key = jax.random.split(jax.random.PRNGKey(seed))

    X = jax.random.uniform(x_key, (n, D), minval=0.5, maxval=1.5)

    def f(x):
        return jnp.sum(x**2) + jnp.sum(x)

    V_raw = jax.random.normal(v_key, (n, D))
    V = V_raw / jnp.linalg.norm(V_raw, axis=1, keepdims=True)

    return SobolevDataset(
        X=X,
        y=jax.vmap(f)(X),
        gradients=jax.vmap(jax.grad(f))(X),
        hvps=jax.vmap(lambda x, v: jax.jvp(jax.grad(f), (x,), (v,))[1])(X, V),
        V=V,
    )


def _config(epochs=1):
    return TrainingConfig(
        learning_rate=1e-2,
        batch_size=16,
        epochs=epochs,
        early_stopping=False,
        print_every=epochs,
        sobolev_order=2,
    )


def _leaf_count(tree, predicate):
    return len(jax.tree_util.tree_leaves(eqx.filter(tree, predicate)))


def test_float_only_model_still_trains():
    dataset = _dataset()

    surrogate = SurrogateModel(FloatOnlyMLP(key=jax.random.PRNGKey(0)))
    trainer = SobolevTrainer(surrogate, _config(epochs=10))

    history = trainer.fit(dataset)

    assert all(math.isfinite(loss) for loss in history["train_loss"])
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_bool_leaf_model_trains():
    dataset = _dataset()

    network = MaskedMLP(jnp.bool_, key=jax.random.PRNGKey(0))

    assert network.mask.dtype == jnp.bool_
    assert eqx.is_array(network.mask)
    assert not eqx.is_inexact_array(network.mask)

    trainer = SobolevTrainer(SurrogateModel(network), _config(epochs=10))

    history = trainer.fit(dataset)

    assert all(math.isfinite(loss) for loss in history["train_loss"])
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_int_leaf_model_trains():
    dataset = _dataset()

    network = MaskedMLP(jnp.int32, key=jax.random.PRNGKey(0))

    assert jnp.issubdtype(network.mask.dtype, jnp.integer)

    trainer = SobolevTrainer(SurrogateModel(network), _config(epochs=10))

    history = trainer.fit(dataset)

    assert all(math.isfinite(loss) for loss in history["train_loss"])
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_optimizer_state_matches_gradient_tree():
    surrogate = SurrogateModel(MaskedMLP(jnp.bool_, key=jax.random.PRNGKey(0)))

    n_all_arrays = _leaf_count(surrogate, eqx.is_array)
    n_inexact = _leaf_count(surrogate, eqx.is_inexact_array)

    assert n_all_arrays == n_inexact + 1, "the bool mask must be the only extra leaf"

    trainer = SobolevTrainer(surrogate, _config())

    opt_state_structure = jax.tree_util.tree_structure(
        trainer.optimizer.init(eqx.filter(surrogate, eqx.is_inexact_array))
    )

    assert jax.tree_util.tree_structure(trainer.opt_state) == opt_state_structure


def test_single_train_step_on_bool_leaf_model():
    dataset = _dataset()

    surrogate = SurrogateModel(MaskedMLP(jnp.bool_, key=jax.random.PRNGKey(0)))
    trainer = SobolevTrainer(surrogate, _config())

    model, opt_state, loss, _ = trainer.train_step(
        surrogate,
        trainer.opt_state,
        dataset.X,
        dataset.y,
        dataset.gradients,
        dataset.hvps,
        dataset.V,
    )

    assert math.isfinite(float(loss))

    weight_before = surrogate.model.mlp.layers[0].weight
    weight_after = model.model.mlp.layers[0].weight

    assert float(jnp.max(jnp.abs(weight_after - weight_before))) > 0.0
    assert bool(jnp.all(model.model.mask == surrogate.model.mask))


def test_non_differentiable_leaf_survives_training():
    dataset = _dataset()

    network = MaskedMLP(jnp.bool_, key=jax.random.PRNGKey(0))
    trainer = SobolevTrainer(SurrogateModel(network), _config(epochs=5))

    trainer.fit(dataset)

    trained_mask = trainer.model.model.mask

    assert trained_mask.dtype == jnp.bool_
    assert bool(jnp.all(trained_mask == network.mask))


if __name__ == "__main__":
    for check in [
        test_float_only_model_still_trains,
        test_bool_leaf_model_trains,
        test_int_leaf_model_trains,
        test_optimizer_state_matches_gradient_tree,
        test_single_train_step_on_bool_leaf_model,
        test_non_differentiable_leaf_survives_training,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
