from typing import Dict, Optional

from risk_visualisierung.riskengine import RiskEngine

from risk_visualisierung.visualizer import Visualizer

from surrogate_modeling.pricing_problem import PricingProblem


def run_xva_analysis(
    surrogate,
    problem: PricingProblem,
    horizon: float = 1.0,
    num_paths: int = 100,
    num_steps: int = 252,
    seed: int = 0,
    min_maturity: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Exposure profiles and XVA along a simulated future."""

    strikes = problem.exposure_strikes()

    if not strikes:
        print(
            f"\nSkipping XVA: '{problem.name}' declares no exposure strikes."
        )
        return None

    simulated = {
        label: problem.exposure_paths(
            strike=strike,
            horizon=horizon,
            num_paths=num_paths,
            num_steps=num_steps,
            seed=seed,
            min_maturity=min_maturity,
        )
        for label, strike in strikes.items()
    }

    if any(paths is None for paths in simulated.values()):
        print(
            f"\nSkipping XVA: '{problem.name}' does not implement "
            f"exposure_paths."
        )
        return None

    print(
        "\nGenerating Monte Carlo paths..."
    )

    time_grid = next(iter(simulated.values()))[0]

    reference_value_fn = _reference_value_fn(problem)

    risk_engine = RiskEngine()

    print("\n===== XVA VALIDATION (across moneyness) =====\n")

    if reference_value_fn is None:
        print(
            f"(no closed form for '{problem.name}': exposures are the "
            f"surrogate's own, with no independent reference)\n"
        )

    results = {}

    for label, strike in strikes.items():

        _, feature_paths = simulated[label]

        xva_k = risk_engine.compute_xva_risk(
            surrogate,
            feature_paths,
            time_grid,
            problem.discount_rate,
        )

        results[label] = {
            "K": strike,
            "xva": xva_k,
            "features": feature_paths,
        }

        if reference_value_fn is None:
            print(
                f"{label:4s} (K={strike:8.2f}) | "
                f"CVA Surrogate: {xva_k['CVA']:.6f} | "
                f"NetXVA Surrogate: {xva_k['NetXVA']:.6f}"
            )
            continue

        xva_reference_k = risk_engine.compute_xva_risk_reference(
            reference_value_fn,
            feature_paths,
            time_grid,
            problem.discount_rate,
        )

        cva_error_k = _relative(xva_k["CVA"], xva_reference_k["CVA"])
        netxva_error_k = _relative(xva_k["NetXVA"], xva_reference_k["NetXVA"])

        print(
            f"{label:4s} (K={strike:8.2f}) | "
            f"CVA Surrogate: {xva_k['CVA']:.6f} | "
            f"CVA Reference: {xva_reference_k['CVA']:.6f} | "
            f"CVA Error: {100*cva_error_k:.2f}% | "
            f"NetXVA Error: {100*netxva_error_k:.2f}%"
        )

        results[label].update(
            {
                "xva_reference": xva_reference_k,
                "cva_error": cva_error_k,
                "netxva_error": netxva_error_k,
            }
        )

    primary = results.get("ATM", results[next(iter(results))])

    xva = primary["xva"]

    Visualizer.plot_mc_paths(
        time_grid,
        primary["features"][:, :, 0],
        num_paths=20,
    )

    Visualizer.report_risk_metrics(
        xva
    )

    _print_primary(primary, reference_value_fn is not None)

    Visualizer.plot_exposure_profiles(
        time_grid,
        xva["EPE"],
        xva["ENE"],
    )

    return xva


def _reference_value_fn(problem: PricingProblem):
    """The problem's closed form, when it has one."""

    if problem.analytic_price(problem.baseline_features()) is None:
        return None

    return problem.analytic_price


def _print_primary(primary, has_reference: bool) -> None:

    xva = primary["xva"]

    print("\n===== XVA (primary strike) =====\n")

    if not has_reference:
        for name in ("CVA", "DVA", "NetXVA"):
            print(f"{name:16s}: {xva[name]:.6f}")
        return

    reference = primary["xva_reference"]

    for name in ("CVA", "DVA", "NetXVA"):
        print(f"{name} Reference : {reference[name]:.6f}")
        print(f"{name} Surrogate : {xva[name]:.6f}")

    print(f"\nCVA Error    : {100*primary['cva_error']:.2f}%")
    print(f"NetXVA Error : {100*primary['netxva_error']:.2f}%")


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / (abs(reference) + 1e-12)
