import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from surrogate_modeling.architectures import (
    ResidualMLP,
    available_architectures,
    build_network,
    register_architecture,
)
from surrogate_modeling.dataset import SobolevDataset
from surrogate_modeling.neuralnetwork import FunctionalNetwork, NeuralNetwork
from surrogate_modeling.sobolev_trainer import SobolevTrainer
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.training_config import TrainingConfig


D = 3


class LinearBasisNet(eqx.Module):
    """Deliberately not an MLP: one linear map over the features [x, x^2]."""

    weights: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size, out_size=1, *, key, **kwargs):
        self.weights = 0.1 * jax.random.normal(key, (out_size, 2 * in_size))
        self.bias = jnp.zeros((out_size,))

    def __call__(self, x):
        return self.weights @ jnp.concatenate([x, x**2]) + self.bias


class AttentionNet(eqx.Module):
    """A different architecture family: self-attention over the features."""

    embed: eqx.nn.Linear
    attention: eqx.nn.MultiheadAttention
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(self, in_size, out_size=1, width_size=16, num_heads=2, *, key, **kwargs):
        embed_key, attention_key, head_key = jax.random.split(key, 3)

        # every input feature (S, K, T, ...) becomes one token
        self.embed = eqx.nn.Linear(1, width_size, key=embed_key)
        self.attention = eqx.nn.MultiheadAttention(num_heads, width_size, key=attention_key)
        self.norm = eqx.nn.LayerNorm(width_size)
        self.head = eqx.nn.Linear(width_size, out_size, key=head_key)

    def __call__(self, x):
        tokens = jax.vmap(self.embed)(x[:, None])

        pooled = jnp.mean(
            self.attention(tokens, tokens, tokens),
            axis=0,
        )

        return self.head(jax.nn.softplus(self.norm(pooled)))


class SelfNormalizingNet(eqx.Module):
    """Brings its own normalization, like diff_ml.nn.Normalized does."""

    seq: eqx.nn.Sequential

    def __init__(self, in_size, x_mean, x_std, y_mean, y_std, *, key):
        self.seq = eqx.nn.Sequential(
            layers=(
                eqx.nn.Lambda(lambda x: (x - x_mean) / x_std),
                eqx.nn.MLP(
                    in_size=in_size,
                    out_size="scalar",
                    width_size=16,
                    depth=2,
                    activation=jax.nn.softplus,
                    key=key,
                ),
                eqx.nn.Lambda(lambda y: y * y_std + y_mean),
            )
        )

    def __call__(self, x, *, key=None):
        return self.seq(x, key=key)


def _params_separate_net(key, in_size, out_size=1, width_size=16):
    """A net in the Flax/Haiku convention: parameters live outside the model."""

    first_key, second_key = jax.random.split(key)

    params = {
        "w1": 0.3 * jax.random.normal(first_key, (width_size, in_size)),
        "b1": jnp.zeros(width_size),
        "w2": 0.3 * jax.random.normal(second_key, (out_size, width_size)),
        "b2": jnp.zeros(out_size),
    }

    def apply_fn(p, x):
        return p["w2"] @ jnp.tanh(p["w1"] @ x + p["b1"]) + p["b2"]

    return params, apply_fn


def _toy_dataset(n=64, seed=0):
    """Sobolev dataset with exact labels for f(x) = sum(x^2) + sum(x)."""

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


def _surrogate(network, dataset):
    return SurrogateModel(
        network,
        jnp.mean(dataset.X, axis=0),
        jnp.std(dataset.X, axis=0),
        jnp.mean(dataset.y),
        jnp.std(dataset.y),
    )


def _fit_and_check(surrogate, dataset, epochs=40):
    config = TrainingConfig(
        learning_rate=1e-2,
        batch_size=16,
        epochs=epochs,
        early_stopping=False,
        print_every=epochs,
        sobolev_order=2,
    )

    trainer = SobolevTrainer(surrogate, config)

    history = trainer.fit(dataset)

    assert all(math.isfinite(loss) for loss in history["train_loss"])
    assert history["train_loss"][-1] < history["train_loss"][0]

    return trainer


def test_builtin_architectures_registered():
    assert "MLP" in available_architectures()
    assert "RESMLP" in available_architectures()


def test_architecture_names_are_case_insensitive():
    key = jax.random.PRNGKey(0)

    lower = build_network("resmlp", key, in_size=D, width_size=8, depth=2)
    upper = build_network("RESMLP", key, in_size=D, width_size=8, depth=2)

    x = jnp.ones(D)

    assert float(jnp.abs(lower(x) - upper(x)).max()) == 0.0


def test_unknown_architecture_raises():
    try:
        build_network("transformer", jax.random.PRNGKey(0), in_size=D)
    except ValueError as e:
        assert "register_architecture" in str(e)
        assert "MLP" in str(e)
    else:
        assert False, "unknown architecture should raise"


def test_duplicate_registration_needs_overwrite():
    try:
        register_architecture("MLP", build_network)
    except ValueError:
        pass
    else:
        assert False, "re-registering a built-in should raise"


def test_custom_architecture_can_be_registered():
    # a plain eqx.Module class is a valid builder
    register_architecture("BASIS", LinearBasisNet, overwrite=True)

    assert "BASIS" in available_architectures()

    network = build_network("BASIS", jax.random.PRNGKey(0), in_size=D)
    dataset = _toy_dataset()

    surrogate = _surrogate(network, dataset)

    assert jnp.isfinite(surrogate.predict_price(dataset.X[0]))
    assert surrogate.predict_gradient(dataset.X[0]).shape == (D,)
    assert surrogate.predict_hessian(dataset.X[0]).shape == (D, D)


def test_surrogate_accepts_raw_equinox_model():
    # goal.txt: "by just accepting JAX models like equinox.nn.MLP"
    network = eqx.nn.MLP(
        in_size=D,
        out_size=1,
        width_size=8,
        depth=2,
        activation=jax.nn.softplus,
        key=jax.random.PRNGKey(0),
    )

    dataset = _toy_dataset()
    surrogate = _surrogate(network, dataset)

    assert surrogate.predict_prices(dataset.X).shape == (len(dataset),)
    assert surrogate.predict_gradients(dataset.X).shape == (len(dataset), D)
    assert surrogate.predict_hvps(dataset.X, dataset.V).shape == (len(dataset), D)


def test_surrogate_accepts_plain_function():
    dataset = _toy_dataset()

    surrogate = _surrogate(lambda x: jnp.sum(x**2), dataset)

    x = dataset.X[0]

    scale = float(surrogate.y_std) / float(surrogate.x_std[0]) ** 2
    expected = 2.0 * float(x[0] - surrogate.x_mean[0]) * scale

    assert abs(float(surrogate.predict_gradient(x)[0]) - expected) < 1e-8


def test_normalization_defaults_to_identity():
    def model(x):
        return jnp.sum(x**2)

    surrogate = SurrogateModel(model)

    x = jnp.array([1.5, -0.5, 2.0])

    assert float(surrogate.predict_price(x)) == float(model(x))


def test_self_normalizing_model_is_not_normalized_twice():
    dataset = _toy_dataset()

    network = SelfNormalizingNet(
        D,
        jnp.mean(dataset.X, axis=0),
        jnp.std(dataset.X, axis=0),
        jnp.mean(dataset.y),
        jnp.std(dataset.y),
        key=jax.random.PRNGKey(0),
    )

    surrogate = SurrogateModel(network)

    x = dataset.X[0]

    assert float(surrogate.predict_price(x)) == float(network(x))

    trainer = _fit_and_check(surrogate, dataset)

    assert math.isfinite(float(trainer.evaluate(dataset)["RMSE"]))


def test_surrogate_rejects_non_callable():
    dataset = _toy_dataset()

    try:
        _surrogate(jnp.zeros(3), dataset)
    except TypeError:
        pass
    else:
        assert False, "a non-callable model should raise"


def test_residual_mlp_is_twice_differentiable():
    # the smooth default activation must survive the skip connections
    network = ResidualMLP(
        in_size=D,
        out_size=1,
        width_size=16,
        depth=4,
        key=jax.random.PRNGKey(0),
    )

    hessian = _surrogate(network, _toy_dataset()).predict_hessian(jnp.ones(D))

    assert bool(jnp.all(jnp.isfinite(hessian)))
    assert float(jnp.max(jnp.abs(hessian))) > 0.0


def test_neural_network_wraps_any_model():
    network = NeuralNetwork(LinearBasisNet(in_size=D, key=jax.random.PRNGKey(0)))

    assert network.architecture == "LinearBasisNet"
    assert jnp.isfinite(network.predict_price(jnp.ones(D)))


def test_neural_network_from_architecture_matches_direct_build():
    key = jax.random.PRNGKey(0)

    wrapped = NeuralNetwork.from_architecture(
        key, in_size=D, architecture="resmlp", width_size=8, depth=2
    )
    direct = build_network("RESMLP", key, in_size=D, width_size=8, depth=2)

    x = jnp.ones(D)

    assert wrapped.architecture == "RESMLP"
    assert float(jnp.abs(wrapped(x) - direct(x)).max()) == 0.0


def test_trainer_runs_with_non_mlp_architecture():
    dataset = _toy_dataset()

    network = build_network(
        "RESMLP", jax.random.PRNGKey(1), in_size=D, width_size=16, depth=2
    )

    trainer = _fit_and_check(_surrogate(network, dataset), dataset)

    assert math.isfinite(float(trainer.evaluate(dataset)["RMSE"]))


def test_foreign_architecture_registers_and_trains():
    register_architecture("ATTENTION", AttentionNet, overwrite=True)

    dataset = _toy_dataset()

    network = build_network("ATTENTION", jax.random.PRNGKey(0), in_size=D, width_size=16)

    trainer = _fit_and_check(_surrogate(network, dataset), dataset)

    # its own weights were optimized, not just some wrapper around it
    moved = jnp.abs(trainer.model.model.head.weight - network.head.weight)

    assert float(jnp.max(moved)) > 0.0


def test_attention_network_is_twice_differentiable():
    # attention/LayerNorm stay smooth, so sobolev_order=2 keeps working
    network = AttentionNet(in_size=D, width_size=16, key=jax.random.PRNGKey(0))

    hessian = _surrogate(network, _toy_dataset()).predict_hessian(jnp.ones(D))

    assert bool(jnp.all(jnp.isfinite(hessian)))
    assert float(jnp.max(jnp.abs(hessian))) > 0.0


def test_functional_network_trains_foreign_parameters():
    # the adapted foreign parameters must still reach the optimizer
    dataset = _toy_dataset()

    params, apply_fn = _params_separate_net(jax.random.PRNGKey(0), D)

    network = FunctionalNetwork(params, apply_fn)

    trainer = _fit_and_check(_surrogate(network, dataset), dataset)

    moved = jnp.abs(trainer.model.model.params["w1"] - params["w1"])

    assert float(jnp.max(moved)) > 0.0


def test_functional_network_rejects_non_callable():
    try:
        FunctionalNetwork({"w": jnp.zeros(3)}, "not a function")
    except TypeError:
        pass
    else:
        assert False, "a non-callable apply_fn should raise"


if __name__ == "__main__":
    for check in [
        test_builtin_architectures_registered,
        test_architecture_names_are_case_insensitive,
        test_unknown_architecture_raises,
        test_duplicate_registration_needs_overwrite,
        test_custom_architecture_can_be_registered,
        test_surrogate_accepts_raw_equinox_model,
        test_surrogate_accepts_plain_function,
        test_normalization_defaults_to_identity,
        test_self_normalizing_model_is_not_normalized_twice,
        test_surrogate_rejects_non_callable,
        test_residual_mlp_is_twice_differentiable,
        test_neural_network_wraps_any_model,
        test_neural_network_from_architecture_matches_direct_build,
        test_trainer_runs_with_non_mlp_architecture,
        test_foreign_architecture_registers_and_trains,
        test_attention_network_is_twice_differentiable,
        test_functional_network_trains_foreign_parameters,
        test_functional_network_rejects_non_callable,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
