import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from surrogate_modeling.dataset import DataLoader, SobolevDataset, train_test_split
from surrogate_modeling.losses import (
    mae_loss,
    mse_loss,
    price_loss,
    relative_l2_error,
    rmse_loss,
    sobolev_loss,
    sobolev_loss_weights,
)
from surrogate_modeling.metrics import (
    mean_relative_error,
    per_dimension_relative_error,
    r2_score,
    smape,
    sobolev_metrics,
)


D = 3


def _dataset(n=10, with_derivatives=True):
    X = jnp.arange(n * D, dtype=jnp.float64).reshape(n, D)
    y = jnp.arange(n, dtype=jnp.float64)

    if not with_derivatives:
        return SobolevDataset(X=X, y=y)

    return SobolevDataset(
        X=X,
        y=y,
        gradients=jnp.ones((n, D)),
        hvps=2.0 * jnp.ones((n, D)),
        V=jnp.ones((n, D)) / jnp.sqrt(D),
    )


def test_dataset_length_and_input_dim():
    dataset = _dataset(7)

    assert len(dataset) == 7
    assert dataset.input_dim == D


def test_get_batch_propagates_missing_derivatives():
    indices = jnp.array([0, 2])

    X, y, grads, hvps, V = _dataset().get_batch(indices)

    assert X.shape == (2, D)
    assert y.shape == (2,)
    assert grads.shape == (2, D)

    X, y, grads, hvps, V = _dataset(with_derivatives=False).get_batch(indices)

    assert grads is None and hvps is None and V is None


def test_dataloader_visits_every_sample_exactly_once():
    dataset = _dataset(10)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    seen = []
    sizes = []

    for X_batch, _, _, _, _ in loader.batches(jax.random.PRNGKey(0)):
        sizes.append(X_batch.shape[0])
        seen.extend(float(row[0]) for row in X_batch)

    assert sizes == [4, 4, 2], "the trailing batch must be shorter, not dropped"
    assert sorted(seen) == sorted(float(row[0]) for row in dataset.X)


def test_dataloader_shuffle_flag_controls_order():
    dataset = _dataset(10)

    def first_column(shuffle, seed):
        loader = DataLoader(dataset, batch_size=10, shuffle=shuffle)
        batch = next(iter(loader.batches(jax.random.PRNGKey(seed))))
        return [float(v) for v in batch[0][:, 0]]

    ordered = first_column(False, 0)

    assert ordered == [float(v) for v in dataset.X[:, 0]]
    assert first_column(True, 0) != ordered
    assert first_column(True, 0) != first_column(True, 1)


def test_train_test_split_partitions_without_overlap():
    dataset = _dataset(10)

    train, test = train_test_split(dataset, train_fraction=0.8)

    assert len(train) == 8
    assert len(test) == 2

    train_rows = {float(row[0]) for row in train.X}
    test_rows = {float(row[0]) for row in test.X}

    assert train_rows.isdisjoint(test_rows)
    assert len(train_rows | test_rows) == 10


def test_train_test_split_keeps_derivatives_optional():
    train, test = train_test_split(_dataset(10, with_derivatives=False))

    assert train.gradients is None and test.gradients is None
    assert train.hvps is None and test.V is None


def test_basic_regression_losses():
    pred = jnp.array([1.0, 2.0, 3.0])
    true = jnp.array([1.0, 2.0, 5.0])

    assert abs(float(mse_loss(pred, true)) - 4.0 / 3.0) < 1e-12
    assert abs(float(rmse_loss(pred, true)) - (4.0 / 3.0) ** 0.5) < 1e-12
    assert abs(float(mae_loss(pred, true)) - 2.0 / 3.0) < 1e-12
    assert float(mse_loss(true, true)) == 0.0


def test_price_loss_reduces_to_mse_at_unit_scale():
    pred = jnp.array([1.0, 2.0, 3.0])
    true = jnp.array([1.5, 2.5, 3.5])

    loss = price_loss(pred, true, scale_floor=1.0, scale_ceiling=1.0)

    assert abs(float(loss) - float(mse_loss(pred, true))) < 1e-12


def test_price_loss_divides_by_the_clipped_scale():
    pred = jnp.array([1.0, 2.0, 3.0])
    true = jnp.array([1.5, 2.5, 3.5])

    loss = price_loss(pred, true, scale_floor=2.0, scale_ceiling=2.0)

    assert abs(float(loss) - float(mse_loss(pred, true)) / 4.0) < 1e-12


def test_price_loss_bounds_the_weight_of_cheap_and_expensive_options():
    cheap = price_loss(jnp.array([0.02]), jnp.array([0.01]),
                       scale_floor=0.05, scale_ceiling=2.0)
    rich = price_loss(jnp.array([1000.01]), jnp.array([1000.0]),
                      scale_floor=0.05, scale_ceiling=2.0)

    ratio = float(cheap) / float(rich)

    assert abs(ratio - (2.0 / 0.05) ** 2) < 1e-6


def test_sobolev_weights_form_a_convex_combination():
    alpha, beta, gamma = sobolev_loss_weights(5)

    assert abs(alpha + beta + gamma - 1.0) < 1e-12
    assert abs(alpha - 1.0 / 11.0) < 1e-12
    assert abs(beta - 5.0 / 11.0) < 1e-12

    alpha, beta, gamma = sobolev_loss_weights(5, use_grad=False)

    assert beta == 0.0
    assert abs(alpha + gamma - 1.0) < 1e-12

    alpha, beta, gamma = sobolev_loss_weights(5, use_grad=False, use_hvp=False)

    assert alpha == 1.0 and beta == 0.0 and gamma == 0.0


def test_sobolev_loss_uses_only_the_terms_supplied():
    prices = jnp.array([1.0, 2.0])

    total, metrics = sobolev_loss(prices, prices, n_dims=D)

    assert float(total) == 0.0
    assert "gradient_loss" not in metrics
    assert "hessian_loss" not in metrics
    assert metrics["alpha"] == 1.0

    grads = jnp.ones((2, D))

    total, metrics = sobolev_loss(
        prices, prices, gradients_pred=grads, gradients_true=2 * grads, n_dims=D
    )

    assert "gradient_loss" in metrics
    assert abs(float(metrics["gradient_loss"]) - 1.0) < 1e-12
    assert abs(float(total) - metrics["beta"] * 1.0) < 1e-12


def test_raw_price_mse_is_reported_but_not_optimized():
    pred = jnp.array([1.0, 2.0])
    true = jnp.array([2.0, 4.0])

    total, metrics = sobolev_loss(
        pred, true, n_dims=D, price_scale_floor=2.0, price_scale_ceiling=2.0
    )

    assert abs(float(metrics["price_mse_raw"]) - float(mse_loss(pred, true))) < 1e-12
    assert float(metrics["price_loss"]) < float(metrics["price_mse_raw"])
    assert abs(float(total) - float(metrics["price_loss"])) < 1e-12


def test_relative_l2_error_endpoints():
    target = jnp.array([3.0, 4.0])

    assert float(relative_l2_error(target, target)) == 0.0
    assert abs(float(relative_l2_error(jnp.zeros(2), target)) - 1.0) < 1e-9


def test_r2_endpoints():
    target = jnp.array([1.0, 2.0, 3.0, 4.0])

    assert abs(float(r2_score(target, target)) - 1.0) < 1e-9

    mean_predictor = jnp.full_like(target, float(jnp.mean(target)))

    assert abs(float(r2_score(mean_predictor, target))) < 1e-9


def test_smape_is_bounded():
    target = jnp.array([1.0, 2.0, 3.0])

    assert float(smape(target, target)) == 0.0
    assert 0.0 <= float(smape(jnp.zeros(3), target)) <= 2.0
    assert float(smape(-target, target)) <= 2.0


def test_mean_relative_error_survives_near_zero_targets():
    target = jnp.array([1e-12, 1.0, 2.0])
    pred = target + 0.01

    assert float(mean_relative_error(pred, target)) < 10.0
    assert bool(jnp.isfinite(mean_relative_error(pred, target)))


def test_per_dimension_error_is_column_wise():
    target = jnp.ones((4, D))
    pred = target.at[:, 1].set(0.0)

    errors = per_dimension_relative_error(pred, target)

    assert errors.shape == (D,)
    assert float(errors[0]) < 1e-9
    assert abs(float(errors[1]) - 1.0) < 1e-9


def test_sobolev_metrics_keys_follow_the_supplied_arguments():
    prices = jnp.array([1.0, 2.0])
    grads = jnp.ones((2, D))

    only_price = sobolev_metrics(prices, prices)

    assert "RMSE" in only_price and "R2" in only_price
    assert "Gradient_RMSE" not in only_price
    assert "HVP_RMSE" not in only_price

    full = sobolev_metrics(prices, prices, grads, grads, grads, grads)

    assert "Gradient_Relative_L2" in full
    assert "HVP_Relative_L2" in full


if __name__ == "__main__":
    for check in [
        test_dataset_length_and_input_dim,
        test_get_batch_propagates_missing_derivatives,
        test_dataloader_visits_every_sample_exactly_once,
        test_dataloader_shuffle_flag_controls_order,
        test_train_test_split_partitions_without_overlap,
        test_train_test_split_keeps_derivatives_optional,
        test_basic_regression_losses,
        test_price_loss_reduces_to_mse_at_unit_scale,
        test_price_loss_divides_by_the_clipped_scale,
        test_price_loss_bounds_the_weight_of_cheap_and_expensive_options,
        test_sobolev_weights_form_a_convex_combination,
        test_sobolev_loss_uses_only_the_terms_supplied,
        test_raw_price_mse_is_reported_but_not_optimized,
        test_relative_l2_error_endpoints,
        test_r2_endpoints,
        test_smape_is_bounded,
        test_mean_relative_error_survives_near_zero_targets,
        test_per_dimension_error_is_column_wise,
        test_sobolev_metrics_keys_follow_the_supplied_arguments,
    ]:
        check()
        print(f"[PASS] {check.__name__}")
