import jax
import jax.numpy as jnp

from typing import Dict, Optional

from surrogate_modeling.pricing_problem import PricingProblem


def run_reference_validation(
    surrogate,
    problem: PricingProblem,
    n_points: int = 256,
    seed: int = 12345,
    arbitrage_tolerance_sigma: float = 3.0,
    label_price_fn=None,
) -> Dict[str, float]:
    """Independent check on a trained surrogate, for any pricing problem."""

    print("\n===== Reference Validation (independent seed) =====\n")

    points = problem.reference_points(n_points=n_points, seed=seed)

    key = jax.random.PRNGKey(seed)

    summary: Dict[str, float] = {}

    reference = _reference_prices(problem, points, key)

    if reference is None:
        print(
            f"No independent Monte Carlo benchmark available for "
            f"'{problem.name}': it does not implement reference_price."
        )
    else:
        predicted = surrogate.predict_prices(points)

        summary.update(_price_report("surrogate", predicted, reference, points))

        if label_price_fn is not None:
            labels = jnp.array(
                [
                    float(label_price_fn(x, jax.random.fold_in(key, 10_000 + i)))
                    for i, x in enumerate(points)
                ]
            )

            summary.update(_price_report("label", labels, reference, points))

        summary.update(_analytic_report(problem, points, reference))

    summary.update(
        _check_arbitrage(surrogate, problem, points, arbitrage_tolerance_sigma)
    )
    summary.update(_check_shape(surrogate, problem, points))
    summary.update(_check_exchangeability(surrogate, problem, points))
    summary.update(_check_comonotonic_limit(problem, points))

    return summary


def _reference_prices(problem, points, key) -> Optional[jnp.ndarray]:
    """High-accuracy re-pricing of every point with an unrelated stream."""

    if problem.reference_price(points[0], key) is None:
        return None

    return jnp.array(
        [
            float(problem.reference_price(x, jax.random.fold_in(key, i)))
            for i, x in enumerate(points)
        ]
    )


def _price_report(label, predicted, reference, points) -> Dict[str, float]:
    """Absolute error, bias and a median relative error."""

    error = predicted - reference

    bias = float(jnp.mean(error))
    rmse = float(jnp.sqrt(jnp.mean(error**2)))
    mae = float(jnp.mean(jnp.abs(error)))

    relative = jnp.abs(error) / jnp.maximum(jnp.abs(reference), 1e-8)
    median_relative = 100 * float(jnp.median(relative))

    smape = 100 * float(
        jnp.mean(
            2.0
            * jnp.abs(error)
            / (jnp.abs(predicted) + jnp.abs(reference) + 1e-8)
        )
    )

    print(
        f"{label:>9s} vs reference | bias {bias:+9.4f} | RMSE {rmse:8.4f} | "
        f"MAE {mae:8.4f} | median rel {median_relative:6.2f}% | SMAPE {smape:6.2f}%"
    )

    return {
        f"{label}_bias": bias,
        f"{label}_rmse": rmse,
        f"{label}_mae": mae,
        f"{label}_median_relative_pct": median_relative,
        f"{label}_smape_pct": smape,
    }


def _analytic_report(problem, points, reference) -> Dict[str, float]:
    """
    Where a closed form exists, how far the Monte Carlo reference sits from it.
    This is the yardstick the surrogate's own error should be read against.
    """

    if problem.analytic_price(points[0]) is None:
        print("           (no closed form for this model)")
        return {}

    analytic = jnp.array([float(problem.analytic_price(x)) for x in points])

    error = reference - analytic

    bias = float(jnp.mean(error))
    rmse = float(jnp.sqrt(jnp.mean(error**2)))

    print(
        f"{'reference':>9s} vs analytic  | bias {bias:+9.4f} | RMSE {rmse:8.4f}"
        f"   <- the noise floor the surrogate is measured against"
    )

    return {"reference_vs_analytic_bias": bias, "reference_vs_analytic_rmse": rmse}


def _check_arbitrage(surrogate, problem, points, tolerance_sigma) -> Dict[str, float]:
    """
    A breach is only reported when it exceeds the Monte Carlo noise of the
    reference itself.
    """

    if problem.arbitrage_bounds(points[0]) is None:
        return {}

    predicted = surrogate.predict_prices(points)

    lower = jnp.array([problem.arbitrage_bounds(x)[0] for x in points])
    upper = jnp.array([problem.arbitrage_bounds(x)[1] for x in points])

    noise = _reference_noise(problem, points)
    tolerance = tolerance_sigma * noise

    breach = jnp.maximum(
        jnp.maximum(lower - predicted, predicted - upper), 0.0
    )

    violated = breach > tolerance

    negative = int(jnp.sum(predicted < 0.0))

    print(
        f"\nNo-arbitrage   : {int(jnp.sum(violated))} of {len(points)} points breach "
        f"by more than {tolerance_sigma:g} sigma of the reference "
        f"({100 * float(jnp.mean(violated)):.1f}%)"
    )
    print(
        f"                 worst breach {float(jnp.max(breach)):.4f}, "
        f"negative prices {negative}"
    )

    return {
        "arbitrage_violation_pct": 100 * float(jnp.mean(violated)),
        "arbitrage_worst_breach": float(jnp.max(breach)),
        "negative_prices": float(negative),
    }


def _reference_noise(problem, points, replications: int = 4) -> jnp.ndarray:
    """Per-point standard error of the reference pricer, measured not assumed."""

    key = jax.random.PRNGKey(24680)

    draws = jnp.stack(
        [
            jnp.array(
                [
                    float(
                        problem.reference_price(
                            x, jax.random.fold_in(jax.random.fold_in(key, j), i)
                        )
                    )
                    for i, x in enumerate(points)
                ]
            )
            for j in range(replications)
        ]
    )

    return jnp.std(draws, axis=0) + 1e-12


def _check_shape(surrogate, problem, points) -> Dict[str, float]:
    """
    Sign conditions the true price gradient satisfies, checked on the
    surrogate. No reference price is needed, so this is the cheapest diagnostic
    here and the hardest to argue with.
    """

    constraints = problem.shape_constraints()

    if not constraints:
        return {}

    gradients = surrogate.predict_gradients(points)

    names = problem.feature_names

    print("\nShape constraints:")

    worst_overall = 0.0
    total_violations = 0

    for constraint in constraints:
        column = gradients[:, names.index(constraint.feature)]

        below = 0.0 if constraint.low is None else jnp.maximum(constraint.low - column, 0.0)
        above = 0.0 if constraint.high is None else jnp.maximum(column - constraint.high, 0.0)

        excess = jnp.maximum(below, above)

        violations = int(jnp.sum(excess > 0.0))
        worst = float(jnp.max(excess))

        total_violations += violations
        worst_overall = max(worst_overall, worst)

        print(
            f"  d/d{constraint.feature:<6s} {violations:5d}/{len(points)} outside "
            f"[{constraint.low}, {constraint.high}]  worst {worst:.5f}   "
            f"({constraint.reason})"
        )

    return {
        "shape_violations": float(total_violations),
        "shape_worst_excess": worst_overall,
    }


def _check_exchangeability(surrogate, problem, points) -> Dict[str, float]:
    """
    Where the true price is invariant under permuting a group of features, the
    surrogate should be too. Reported in absolute terms and as a median
    relative deviation - the previous worst-relative figure was set by a single
    near-zero price.
    """

    indices = problem.exchangeable_features

    if len(indices) < 2:
        return {}

    order = jnp.array(indices)
    rolled = jnp.roll(order, 1)

    permuted = points.at[:, order].set(points[:, rolled])

    base = surrogate.predict_prices(points)
    other = surrogate.predict_prices(permuted)

    deviation = jnp.abs(other - base)
    relative = deviation / jnp.maximum(jnp.abs(base), 1e-8)

    names = ", ".join(problem.feature_names[i] for i in indices)

    print(
        f"\nPermutation symmetry over ({names}): mean {float(jnp.mean(deviation)):.4f}, "
        f"worst {float(jnp.max(deviation)):.4f} in price units; "
        f"median {100 * float(jnp.median(relative)):.3f}% relative"
    )

    return {
        "permutation_mean_deviation": float(jnp.mean(deviation)),
        "permutation_worst_deviation": float(jnp.max(deviation)),
        "permutation_median_relative_pct": 100 * float(jnp.median(relative)),
    }


def _check_comonotonic_limit(problem, points) -> Dict[str, float]:
    """
    The one testable consequence of an assumed dependence structure:
    diversification can only reduce a call's value, so the simulated price must
    not exceed its perfectly-correlated limit. Skipped where the problem
    declares no such limit.
    """

    if problem.comonotonic_limit_price(points[0]) is None:
        return {}

    n_assets = len(problem.exchangeable_features) or 1

    collapsed = points.at[:, :n_assets].set(
        jnp.repeat(jnp.mean(points[:, :n_assets], axis=1, keepdims=True), n_assets, axis=1)
    )

    key = jax.random.PRNGKey(97531)

    simulated = jnp.array(
        [
            float(problem.reference_price(x, jax.random.fold_in(key, i)))
            for i, x in enumerate(collapsed[:64])
        ]
    )

    analytic = jnp.array(
        [float(problem.comonotonic_limit_price(x)) for x in collapsed[:64]]
    )

    excess = simulated - analytic

    print(
        f"\nDiversification: basket minus its rho=1 limit, mean "
        f"{float(jnp.mean(excess)):+.4f}, max {float(jnp.max(excess)):+.4f} "
        f"(must be <= 0 for a call)"
    )

    return {
        "diversification_mean_excess": float(jnp.mean(excess)),
        "diversification_max_excess": float(jnp.max(excess)),
    }
