import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from surrogate_modeling.dataset import SobolevDataset, train_test_split
from surrogate_modeling.sobolev_trainer import SobolevTrainer
from surrogate_modeling.surrogate_model import SurrogateModel
from surrogate_modeling.training_config import TrainingConfig


D = 3

HISTORY_KEYS = [
    "train_loss", "valid_loss",
    "train_price_rmse", "valid_price_rmse",
    "train_price_loss", "train_gradient_loss", "train_hessian_loss",
    "valid_price_loss", "valid_gradient_loss", "valid_hessian_loss",
]


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


def _surrogate(dataset, seed=0):
    network = eqx.nn.MLP(
        D, "scalar", 16, 2, activation=jax.nn.softplus, key=jax.random.PRNGKey(seed)
    )

    return SurrogateModel(
        network,
        jnp.mean(dataset.X, axis=0),
        jnp.std(dataset.X, axis=0),
        jnp.mean(dataset.y),
        jnp.std(dataset.y),
    )


def _config(**overrides):
    base = dict(
        learning_rate=1e-2, batch_size=16, epochs=10,
        early_stopping=False, print_every=1000, sobolev_order=2,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def test_history_has_all_keys_with_matching_lengths():
    dataset = _dataset()
    train, valid = train_test_split(dataset)

    history = SobolevTrainer(_surrogate(train), _config(epochs=4)).fit(train, valid)

    for key in HISTORY_KEYS:
        assert key in history, key
        assert len(history[key]) == 4, key


def test_without_validation_only_train_curves_are_filled():
    dataset = _dataset()

    history = SobolevTrainer(_surrogate(dataset), _config(epochs=3)).fit(dataset)

    assert len(history["train_loss"]) == 3
    assert history["valid_loss"] == []
    assert history["valid_price_rmse"] == []


def test_early_stopping_halts_when_validation_stops_improving():
    train, valid = train_test_split(_dataset())

    config = _config(epochs=50, early_stopping=True, patience=2, min_delta=1e9)

    history = SobolevTrainer(_surrogate(train), config).fit(train, valid)

    # epoch 0 always "improves" (best starts at inf), then patience runs out
    assert len(history["train_loss"]) == 3


def test_early_stopping_disabled_runs_every_epoch():
    train, valid = train_test_split(_dataset())

    config = _config(epochs=6, early_stopping=False, patience=2, min_delta=1e9)

    history = SobolevTrainer(_surrogate(train), config).fit(train, valid)

    assert len(history["train_loss"]) == 6


def test_best_model_is_restored_after_fit():
    train, valid = train_test_split(_dataset())

    trainer = SobolevTrainer(_surrogate(train), _config(epochs=8, min_delta=0.0))
    history = trainer.fit(train, valid)

    final_loss, _ = trainer.compute_loss(
        trainer.model, valid.X, valid.y, valid.gradients, valid.hvps, valid.V
    )

    assert abs(float(final_loss) - min(history["valid_loss"])) < 1e-9


def test_checkpoint_file_is_written():
    train, valid = train_test_split(_dataset())

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "best_model.eqx")

        trainer = SobolevTrainer(
            _surrogate(train), _config(epochs=3, min_delta=0.0), checkpoint_path=path
        )
        trainer.fit(train, valid)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        restored = eqx.tree_deserialise_leaves(path, trainer.model)

        assert bool(
            jnp.allclose(
                restored.predict_prices(valid.X), trainer.model.predict_prices(valid.X)
            )
        )


def test_sobolev_order_gates_the_loss_terms():
    dataset = _dataset()

    orders = {}
    for order in (0, 1, 2):
        history = SobolevTrainer(
            _surrogate(dataset), _config(epochs=2, sobolev_order=order)
        ).fit(dataset)

        orders[order] = (
            history["train_gradient_loss"][0],
            history["train_hessian_loss"][0],
        )

    assert orders[0] == (0.0, 0.0)
    assert orders[1][0] > 0.0 and orders[1][1] == 0.0
    assert orders[2][0] > 0.0 and orders[2][1] > 0.0


def test_greek_rescaling_changes_the_loss():
    dataset = _dataset()
    surrogate = _surrogate(dataset)

    scale = surrogate.x_std / surrogate.y_std

    grad_scale = jnp.maximum(jnp.std(dataset.gradients * scale, axis=0), 1e-6)
    hvp_scale = jnp.maximum(jnp.std(dataset.hvps * scale, axis=0), 1e-6)

    plain = SobolevTrainer(surrogate, _config())
    scaled = SobolevTrainer(
        surrogate, _config(), grad_scale=grad_scale, hvp_scale=hvp_scale
    )

    args = (surrogate, dataset.X, dataset.y, dataset.gradients, dataset.hvps, dataset.V)

    unscaled_loss, _ = plain.compute_loss(*args)
    scaled_loss, _ = scaled.compute_loss(*args)

    assert math.isfinite(float(unscaled_loss))
    assert math.isfinite(float(scaled_loss))
    assert float(unscaled_loss) != float(scaled_loss)


def test_evaluate_reports_price_gradient_and_hvp_metrics():
    dataset = _dataset()

    trainer = SobolevTrainer(_surrogate(dataset), _config(epochs=2))
    trainer.fit(dataset)

    metrics = trainer.evaluate(dataset)

    for key in ["RMSE", "MAE", "R2", "Gradient_RMSE", "HVP_RMSE"]:
        assert key in metrics, key
        assert math.isfinite(float(metrics[key])), key


def test_evaluate_skips_absent_derivative_metrics():
    dataset = _dataset()
    price_only = SobolevDataset(X=dataset.X, y=dataset.y)

    trainer = SobolevTrainer(_surrogate(dataset), _config(epochs=1))

    metrics = trainer.evaluate(price_only)

    assert "RMSE" in metrics
    assert "Gradient_RMSE" not in metrics
    assert "HVP_RMSE" not in metrics


def test_compute_loss_is_deterministic():
    dataset = _dataset()
    surrogate = _surrogate(dataset)
    trainer = SobolevTrainer(surrogate, _config())

    args = (surrogate, dataset.X, dataset.y, dataset.gradients, dataset.hvps, dataset.V)

    first, _ = trainer.compute_loss(*args)
    second, _ = trainer.compute_loss(*args)

    assert float(first) == float(second)


def test_training_config_rejects_invalid_values():
    invalid = [
        dict(learning_rate=0.0),
        dict(learning_rate=-1e-3),
        dict(batch_size=0),
        dict(epochs=0),
        dict(sobolev_order=3),
        dict(sobolev_order=-1),
        dict(lambda_grad=-1.0),
        dict(lambda_hessian=-1.0),
    ]

    for overrides in invalid:
        try:
            TrainingConfig(**overrides).validate()
        except ValueError:
            pass
        else:
            assert False, f"{overrides} should be rejected"

    TrainingConfig().validate()


if __name__ == "__main__":
    for check in [
        test_history_has_all_keys_with_matching_lengths,
        test_without_validation_only_train_curves_are_filled,
        test_early_stopping_halts_when_validation_stops_improving,
        test_early_stopping_disabled_runs_every_epoch,
        test_best_model_is_restored_after_fit,
        test_checkpoint_file_is_written,
        test_sobolev_order_gates_the_loss_terms,
        test_greek_rescaling_changes_the_loss,
        test_evaluate_reports_price_gradient_and_hvp_metrics,
        test_evaluate_skips_absent_derivative_metrics,
        test_compute_loss_is_deterministic,
        test_training_config_rejects_invalid_values,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
