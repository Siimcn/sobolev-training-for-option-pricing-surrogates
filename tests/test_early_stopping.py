import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from surrogate_modeling.dataset import SobolevDataset, train_test_split
from surrogate_modeling.sobolev_trainer import SobolevTrainer
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.training_config import (
    PRICE_GRADIENT,
    SELECTION_METRICS,
    TOTAL,
    TrainingConfig,
)


D = 3


def _dataset(n=64, seed=0):
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


def _trainer(dataset, **overrides):
    base = dict(
        learning_rate=1e-2, batch_size=16, epochs=10,
        early_stopping=False, print_every=1000, sobolev_order=2,
    )
    base.update(overrides)

    network = eqx.nn.MLP(
        D, "scalar", 16, 2, activation=jax.nn.softplus, key=jax.random.PRNGKey(0)
    )

    surrogate = SurrogateModel(
        network,
        jnp.mean(dataset.X, axis=0),
        jnp.std(dataset.X, axis=0),
        jnp.mean(dataset.y),
        jnp.std(dataset.y),
    )

    return SobolevTrainer(surrogate, TrainingConfig(**base))


def test_defaults_keep_the_previous_behaviour():
    config = TrainingConfig()

    assert config.selection_metric == TOTAL
    assert config.min_delta_relative == 0.0

    config.validate()


def test_config_rejects_invalid_selection_settings():
    for overrides in [
        dict(selection_metric="hvp_only"),
        dict(min_delta_relative=-0.1),
    ]:
        try:
            TrainingConfig(**overrides).validate()
        except ValueError:
            pass
        else:
            assert False, f"{overrides} should be rejected"

    for metric in SELECTION_METRICS:
        TrainingConfig(selection_metric=metric).validate()


def test_absolute_threshold_is_used_on_the_first_epoch():
    # best_loss starts at inf, so a relative threshold would be infinite
    trainer = _trainer(_dataset(), min_delta=1e-6, min_delta_relative=1e-3)

    assert trainer._improvement_threshold(float("inf")) == 1e-6
    assert abs(trainer._improvement_threshold(0.5) - 5e-4) < 1e-15


def test_relative_threshold_scales_with_the_loss():
    trainer = _trainer(_dataset(), min_delta_relative=1e-2)

    # a fixed absolute threshold would demand the same gain at every scale
    assert abs(trainer._improvement_threshold(1.0) - 1e-2) < 1e-15
    assert abs(trainer._improvement_threshold(0.01) - 1e-4) < 1e-15


def test_absolute_threshold_when_relative_is_off():
    trainer = _trainer(_dataset(), min_delta=1e-4, min_delta_relative=0.0)

    assert trainer._improvement_threshold(1.0) == 1e-4
    assert trainer._improvement_threshold(1e-9) == 1e-4


def test_price_gradient_selection_drops_the_hvp_term():
    trainer = _trainer(_dataset(), selection_metric=PRICE_GRADIENT)

    metrics = {
        "price_loss": 2.0,
        "gradient_loss": 4.0,
        "hessian_loss": 1000.0,
        "alpha": 0.2,
        "beta": 0.6,
        "gamma": 0.2,
    }

    selection = trainer._selection_loss(999.0, metrics)

    expected = (0.2 * 2.0 + 0.6 * 4.0) / (0.2 + 0.6)

    assert abs(selection - expected) < 1e-12

    # the huge HVP term must not move it at all
    metrics["hessian_loss"] = 1e9
    assert abs(trainer._selection_loss(999.0, metrics) - expected) < 1e-12


def test_total_selection_uses_the_objective_unchanged():
    trainer = _trainer(_dataset(), selection_metric=TOTAL)

    metrics = {"price_loss": 2.0, "gradient_loss": 4.0, "alpha": 0.2, "beta": 0.6}

    assert trainer._selection_loss(7.5, metrics) == 7.5


def test_price_gradient_selection_survives_a_missing_gradient_term():
    trainer = _trainer(_dataset(), selection_metric=PRICE_GRADIENT)

    metrics = {"price_loss": 3.0, "alpha": 0.5, "beta": 0.5}

    assert abs(trainer._selection_loss(9.0, metrics) - 3.0) < 1e-12


def test_relative_threshold_stops_a_converged_run():
    # a threshold this coarse cannot be met, so patience runs out at once
    train, valid = train_test_split(_dataset())

    history = _trainer(
        train,
        epochs=50,
        early_stopping=True,
        patience=2,
        min_delta_relative=10.0,
    ).fit(train, valid)

    assert len(history["train_loss"]) == 3


def test_training_still_learns_with_the_new_selection():
    train, valid = train_test_split(_dataset())

    history = _trainer(
        train,
        epochs=30,
        early_stopping=True,
        patience=50,
        min_delta_relative=1e-3,
        selection_metric=PRICE_GRADIENT,
    ).fit(train, valid)

    assert len(history["train_loss"]) == 30
    assert history["train_loss"][-1] < history["train_loss"][0]


if __name__ == "__main__":
    for check in [
        test_defaults_keep_the_previous_behaviour,
        test_config_rejects_invalid_selection_settings,
        test_absolute_threshold_is_used_on_the_first_epoch,
        test_relative_threshold_scales_with_the_loss,
        test_absolute_threshold_when_relative_is_off,
        test_price_gradient_selection_drops_the_hvp_term,
        test_total_selection_uses_the_objective_unchanged,
        test_price_gradient_selection_survives_a_missing_gradient_term,
        test_relative_threshold_stops_a_converged_run,
        test_training_still_learns_with_the_new_selection,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
