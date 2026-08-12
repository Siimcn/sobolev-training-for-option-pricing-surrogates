import jax.numpy as jnp
from typing import Dict


def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y_pred - y_true) ** 2)


def rmse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(mse(y_pred, y_true))


def mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_pred - y_true))


def relative_l2_error(
    prediction: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-12
) -> jnp.ndarray:

    numerator = jnp.linalg.norm(prediction - target)

    denominator = jnp.linalg.norm(target) + eps

    return numerator / denominator


def mean_relative_error(
    prediction: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-12
) -> jnp.ndarray:
    """
    Denominator floored at a small fraction of the average target, so
    near-zero targets (deep OTM prices) do not blow this up. Prefer
    SMAPE when that matters.
    """

    scale_floor = jnp.maximum(eps, 1e-3 * jnp.mean(jnp.abs(target)))

    denominator = jnp.maximum(jnp.abs(target), scale_floor)

    return jnp.mean(jnp.abs(prediction - target) / denominator)


def smape(
    prediction: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """
    Symmetric mean absolute percentage error. Unlike
    `mean_relative_error` it stays bounded for near-zero prices.
    """

    return jnp.mean(
        2.0
        * jnp.abs(prediction - target)
        / (jnp.abs(prediction) + jnp.abs(target) + eps)
    )


def per_dimension_relative_error(
    prediction: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-12
) -> jnp.ndarray:
    """
    Relative L2 error per column. A pooled norm over the whole (N, d)
    array can hide that some Greeks (delta) are learned well while
    others (vega, rho) are not. Not part of `sobolev_metrics`.
    """

    num = jnp.linalg.norm(prediction - target, axis=0)
    den = jnp.linalg.norm(target, axis=0) + eps

    return num / den


def r2_score(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:

    ss_res = jnp.sum((prediction - target) ** 2)

    ss_tot = jnp.sum((target - jnp.mean(target)) ** 2)

    return 1.0 - ss_res / (ss_tot + 1e-12)


def price_metrics(
    prices_pred: jnp.ndarray, prices_true: jnp.ndarray
) -> Dict[str, float]:

    return {
        "RMSE": float(rmse(prices_pred, prices_true)),
        "MAE": float(mae(prices_pred, prices_true)),
        "Relative_L2": float(relative_l2_error(prices_pred, prices_true)),
        "SMAPE": float(smape(prices_pred, prices_true)),
        "Mean_Relative_Error": float(mean_relative_error(prices_pred, prices_true)),
        "R2": float(r2_score(prices_pred, prices_true)),
    }


def gradient_metrics(
    gradients_pred: jnp.ndarray, gradients_true: jnp.ndarray
) -> Dict[str, float]:

    return {
        "Gradient_RMSE": float(rmse(gradients_pred, gradients_true)),
        "Gradient_MAE": float(mae(gradients_pred, gradients_true)),
        "Gradient_Relative_L2": float(
            relative_l2_error(gradients_pred, gradients_true)
        ),
    }


def hvp_metrics(hvps_pred: jnp.ndarray, hvps_true: jnp.ndarray) -> Dict[str, float]:

    return {
        "HVP_RMSE": float(rmse(hvps_pred, hvps_true)),
        "HVP_MAE": float(mae(hvps_pred, hvps_true)),
        "HVP_Relative_L2": float(relative_l2_error(hvps_pred, hvps_true)),
    }


def sobolev_metrics(
    prices_pred: jnp.ndarray,
    prices_true: jnp.ndarray,
    gradients_pred: jnp.ndarray | None = None,
    gradients_true: jnp.ndarray | None = None,
    hvps_pred: jnp.ndarray | None = None,
    hvps_true: jnp.ndarray | None = None,
) -> Dict[str, float]:

    metrics = {}

    metrics.update(price_metrics(prices_pred, prices_true))

    if gradients_pred is not None and gradients_true is not None:
        metrics.update(gradient_metrics(gradients_pred, gradients_true))

    if hvps_pred is not None and hvps_true is not None:
        metrics.update(hvp_metrics(hvps_pred, hvps_true))

    return metrics
