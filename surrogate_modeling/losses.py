import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple


def mse_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((prediction - target) ** 2)


def rmse_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(mse_loss(prediction, target))


def mae_loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(prediction - target))


def price_loss(
    prices_pred: jnp.ndarray,
    prices_true: jnp.ndarray,
    scale_floor: Optional[jnp.ndarray] = None,
    scale_ceiling: Optional[jnp.ndarray] = None,
    floor_fraction: float = 0.3,
    ceiling_fraction: float = 2.0,
    min_floor: float = 0.05,
) -> jnp.ndarray:
    """
    Relative-residual price loss: mean(((pred-true)/scale)^2), where `scale` is
    |true price| clipped to [scale_floor, scale_ceiling].
    """
    abs_true = jnp.abs(prices_true)

    if scale_floor is None or scale_ceiling is None:
        local_std = jnp.std(prices_true) + 1e-8
        if scale_floor is None:
            scale_floor = jnp.maximum(min_floor, floor_fraction * local_std)
        if scale_ceiling is None:
            scale_ceiling = jnp.maximum(scale_floor * 2.0, ceiling_fraction * local_std)

    scale = jnp.clip(abs_true, scale_floor, scale_ceiling)

    return jnp.mean(((prices_pred - prices_true) / scale) ** 2)


def gradient_loss(
    gradients_pred: jnp.ndarray, gradients_true: jnp.ndarray
) -> jnp.ndarray:
    return mse_loss(gradients_pred, gradients_true)


def hvp_loss(hvps_pred: jnp.ndarray, hvps_true: jnp.ndarray) -> jnp.ndarray:
    return mse_loss(hvps_pred, hvps_true)


def sobolev_loss_weights(
    n_dims: int,
    grad_weight: float = 1.0,
    hvp_weight: float = 1.0,
    use_grad: bool = True,
    use_hvp: bool = True,
) -> Tuple[float, float, float]:
    """
    Returns convex-combination weights (alpha, beta, gamma) with alpha + beta +
    gamma == 1.
    """
    grad_scale = grad_weight * n_dims if use_grad else 0.0
    hvp_scale = hvp_weight * n_dims if use_hvp else 0.0

    denom = 1.0 + grad_scale + hvp_scale

    alpha = 1.0 / denom
    beta = grad_scale / denom
    gamma = hvp_scale / denom

    return alpha, beta, gamma


def sobolev_loss(
    prices_pred: jnp.ndarray,
    prices_true: jnp.ndarray,
    gradients_pred: Optional[jnp.ndarray] = None,
    gradients_true: Optional[jnp.ndarray] = None,
    hvps_pred: Optional[jnp.ndarray] = None,
    hvps_true: Optional[jnp.ndarray] = None,
    n_dims: int = 1,
    lambda_grad: float = 1.0,
    lambda_hessian: float = 1.0,
    price_scale_floor: Optional[jnp.ndarray] = None,
    price_scale_ceiling: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    L = alpha * L_price + beta * L_grad + gamma * L_hvp, alpha + beta + gamma =
    1 (over the terms actually present).
    """

    use_grad = gradients_pred is not None and gradients_true is not None
    use_hvp = hvps_pred is not None and hvps_true is not None

    alpha, beta, gamma = sobolev_loss_weights(
        n_dims,
        grad_weight=lambda_grad,
        hvp_weight=lambda_hessian,
        use_grad=use_grad,
        use_hvp=use_hvp,
    )

    lp = price_loss(
        prices_pred,
        prices_true,
        scale_floor=price_scale_floor,
        scale_ceiling=price_scale_ceiling,
    )

    total = alpha * lp

    metrics = {
        "price_loss": lp,
        "price_mse_raw": mse_loss(prices_pred, prices_true),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    if use_grad:
        lg = gradient_loss(gradients_pred, gradients_true)
        total += beta * lg
        metrics["gradient_loss"] = lg

    if use_hvp:
        lh = hvp_loss(hvps_pred, hvps_true)
        total += gamma * lh
        metrics["hessian_loss"] = lh

    metrics["total_loss"] = total

    return total, metrics


def relative_l2_error(
    prediction: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-12
) -> jnp.ndarray:
    numerator = jnp.linalg.norm(prediction - target)

    denominator = jnp.linalg.norm(target) + eps

    return numerator / denominator
