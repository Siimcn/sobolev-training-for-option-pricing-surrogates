import jax
import jax.numpy as jnp

from typing import Dict, Optional

from surrogate_modeling.pricing_problem import PricingProblem


def run_reference_validation(
    surrogate,
    problem: PricingProblem,
    n_points: int = 9,
    seed: int = 12345,
) -> Dict[str, float]:
    """
    Independent check on a trained surrogate, for any pricing problem.

    Three things the training metrics cannot tell you:

    - the surrogate is re-priced against Monte Carlo run with a seed
      unrelated to the labels, so agreement is not just memorised noise;
    - where the model has a closed form, MC is compared against it too,
      which separates "learned the MC operator" from "learned the
      formula" - the two are close by construction, so the surrogate
      being no further from MC than MC is from its own benchmark is the
      meaningful statement;
    - model-free properties - no-arbitrage bounds and any exchangeability
      the problem declares - are checked on the surrogate directly.

    Everything comes from `problem`, so a new model is validated the same
    way without adding code here. Checks a problem does not support are
    reported as unavailable rather than silently skipped.
    """

    print("\n===== Reference Validation (independent seed) =====\n")

    points = problem.reference_points(n_points=n_points, seed=seed)

    key = jax.random.PRNGKey(seed)

    rows = _reference_table(surrogate, problem, points, key)

    summary: Dict[str, float] = {}

    if rows:
        summary.update(_summarise(rows))
    else:
        print(
            f"No independent Monte Carlo benchmark available for "
            f"'{problem.name}': it does not implement reference_price."
        )

    summary.update(_check_arbitrage_bounds(surrogate, problem, points))
    summary.update(_check_exchangeability(surrogate, problem, points))

    return summary


def _reference_table(surrogate, problem, points, key):
    """Surrogate vs fresh Monte Carlo vs closed form, one row per point."""

    names = problem.feature_names

    feature_header = " ".join(f"{name:>9s}" for name in names)

    has_analytic = problem.analytic_price(points[0]) is not None

    analytic_header = f" {'Analytic':>10s}" if has_analytic else ""
    analytic_error_header = f" {'MC-vs-Ref':>10s}" if has_analytic else ""

    rows = []

    for i, x in enumerate(points):

        fresh = problem.reference_price(x, jax.random.fold_in(key, i))

        if fresh is None:
            return []

        if not rows:
            print(
                f"{feature_header} | {'Surrogate':>10s} {'Fresh MC':>10s}"
                f"{analytic_header} | {'Sur-vs-MC':>10s}{analytic_error_header}"
            )

        surrogate_price = float(surrogate.predict_price(x))
        fresh_price = float(fresh)

        analytic = problem.analytic_price(x)
        analytic_price = None if analytic is None else float(analytic)

        surrogate_vs_mc = _relative(surrogate_price, fresh_price)

        mc_vs_analytic = (
            None
            if analytic_price is None
            else _relative(fresh_price, analytic_price)
        )

        feature_values = " ".join(f"{float(v):9.4f}" for v in x)

        analytic_cell = (
            "" if analytic_price is None else f" {analytic_price:10.4f}"
        )

        analytic_error_cell = (
            "" if mc_vs_analytic is None else f" {100 * mc_vs_analytic:9.2f}%"
        )

        print(
            f"{feature_values} | {surrogate_price:10.4f} {fresh_price:10.4f}"
            f"{analytic_cell} | {100 * surrogate_vs_mc:9.2f}%{analytic_error_cell}"
        )

        rows.append(
            {
                "surrogate": surrogate_price,
                "reference": fresh_price,
                "analytic": analytic_price,
                "surrogate_vs_reference": surrogate_vs_mc,
                "reference_vs_analytic": mc_vs_analytic,
            }
        )

    return rows


def _summarise(rows) -> Dict[str, float]:

    mean_surrogate = 100 * _mean(row["surrogate_vs_reference"] for row in rows)

    print(
        f"\nMean |surrogate - fresh MC| / |fresh MC|   : {mean_surrogate:.2f}%"
    )

    summary = {"MeanSurrogateVsReference_pct": mean_surrogate}

    analytic_errors = [
        row["reference_vs_analytic"]
        for row in rows
        if row["reference_vs_analytic"] is not None
    ]

    if not analytic_errors:
        print(
            "(no closed form for this model, so the Monte Carlo benchmark "
            "stands alone)"
        )
        return summary

    mean_analytic = 100 * _mean(analytic_errors)

    print(
        f"Mean |fresh MC  - analytic| / |analytic|   : {mean_analytic:.2f}%"
    )

    print(
        "(if the first number is not clearly larger than the second, the surrogate\n"
        " is at least as consistent with Monte Carlo as MC is with its own analytic\n"
        " benchmark - i.e. there is no evidence the surrogate learned the closed\n"
        " form instead of the MC operator.)"
    )

    summary["MeanReferenceVsAnalytic_pct"] = mean_analytic

    return summary


def _check_arbitrage_bounds(surrogate, problem, points) -> Dict[str, float]:
    """
    Model-free: a call is worth at least its discounted intrinsic value
    and never more than the underlying.
    """

    if problem.arbitrage_bounds(points[0]) is None:
        return {}

    violations = 0
    worst = 0.0

    for x in points:
        lower, upper = problem.arbitrage_bounds(x)

        price = float(surrogate.predict_price(x))

        excess = max(lower - price, price - upper, 0.0)

        if excess > 0.0:
            violations += 1
            worst = max(worst, excess)

    scale = float(jnp.mean(jnp.abs(jnp.array([
        float(surrogate.predict_price(x)) for x in points
    ])))) + 1e-12

    print(
        f"\nNo-arbitrage bounds : {len(points) - violations}/{len(points)} points inside"
        + (f", worst breach {worst:.4f} ({100 * worst / scale:.2f}% of mean price)" if violations else "")
    )

    return {
        "ArbitrageViolations": float(violations),
        "WorstArbitrageBreach": worst,
    }


def _check_exchangeability(surrogate, problem, points) -> Dict[str, float]:
    """
    Where the true price is invariant under permuting a group of features,
    the surrogate should be too. Nothing enforces it architecturally, so
    the residual asymmetry is a real error the pooled metrics hide.
    """

    indices = problem.exchangeable_features

    if len(indices) < 2:
        return {}

    order = jnp.array(indices)
    rolled = jnp.roll(order, 1)

    worst = 0.0

    for x in points:
        permuted = x.at[order].set(x[rolled])

        base = float(surrogate.predict_price(x))
        other = float(surrogate.predict_price(permuted))

        worst = max(worst, _relative(other, base))

    names = ", ".join(problem.feature_names[i] for i in indices)

    print(
        f"Permutation symmetry: worst deviation {100 * worst:.4f}% "
        f"over ({names})"
    )

    return {"WorstPermutationDeviation_pct": 100 * worst}


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / (abs(reference) + 1e-8)


def _mean(values) -> float:
    values = list(values)

    return sum(values) / len(values)
