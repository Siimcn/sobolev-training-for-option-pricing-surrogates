import jax
import jax.numpy as jnp

from marktsimulation.pricing_model import (
    BlackScholesModel,
    BlackScholesParams,
)

from marktsimulation.timesteppingscheme import (
    EulerMaruyama,
)

from marktsimulation.black_scholes_mc import (
    bs_mc_price,
)

from marktsimulation.black_scholes import (
    black_scholes_price_single,
)

from risk_visualisierung.riskengine import (
    RiskEngine,
)

from risk_visualisierung.visualizer import (
    Visualizer,
)


def mc_reference_check(
    surrogate,
    market_data,
    fitted_params,
    seed: int = 12345,
):
    """
    Independent Monte-Carlo-vs-analytic-vs-surrogate sanity check.

    Comparing the surrogate against the analytic BS price alone cannot
    tell "learned the MC operator" apart from "learned the analytic
    formula", since the two are close by construction. This re-runs the
    MC pricer with a fresh seed (training uses a fixed PRNGKey(0)) and
    reports all three side by side.
    """

    print("\n===== Monte Carlo Reference Check (independent seed) =====\n")

    spot = market_data.spot
    sigma = fitted_params.sigma
    r = fitted_params.r

    strikes = [0.85 * spot, spot, 1.15 * spot]
    maturities = [0.1, 0.5, 1.0]

    key = jax.random.PRNGKey(seed)

    print(f"{'S':>8s} {'K':>8s} {'T':>6s} | {'Surrogate':>10s} {'Fresh MC':>10s} {'Analytic':>10s} | {'Sur-vs-MC':>10s} {'MC-vs-BS':>10s}")

    rows = []

    for i, (K, T) in enumerate([(K, T) for K in strikes for T in maturities]):

        x = jnp.array([spot, K, T, sigma, r])

        surrogate_price = float(surrogate.predict_price(x))

        mc_key = jax.random.fold_in(key, i)
        fresh_mc_price = float(bs_mc_price(x, key=mc_key))

        analytic_price = float(
            black_scholes_price_single(
                spot=spot, strike=K, maturity=max(T, 1e-8), sigma=sigma, r=r, is_call=True
            )
        )

        sur_vs_mc = abs(surrogate_price - fresh_mc_price) / (abs(fresh_mc_price) + 1e-8)
        mc_vs_bs = abs(fresh_mc_price - analytic_price) / (abs(analytic_price) + 1e-8)

        print(
            f"{spot:8.2f} {K:8.2f} {T:6.3f} | "
            f"{surrogate_price:10.4f} {fresh_mc_price:10.4f} {analytic_price:10.4f} | "
            f"{100*sur_vs_mc:9.2f}% {100*mc_vs_bs:9.2f}%"
        )

        rows.append(
            {
                "S": spot, "K": K, "T": T,
                "surrogate": surrogate_price, "mc": fresh_mc_price, "analytic": analytic_price,
                "surrogate_vs_mc_pct": 100 * sur_vs_mc, "mc_vs_analytic_pct": 100 * mc_vs_bs,
            }
        )

    mean_sur_vs_mc = sum(row["surrogate_vs_mc_pct"] for row in rows) / len(rows)
    mean_mc_vs_bs = sum(row["mc_vs_analytic_pct"] for row in rows) / len(rows)

    print(
        f"\nMean |surrogate - fresh MC| / |fresh MC|   : {mean_sur_vs_mc:.2f}%"
    )
    print(
        f"Mean |fresh MC  - analytic| / |analytic|   : {mean_mc_vs_bs:.2f}%"
    )
    print(
        "(if the first number is not clearly larger than the second, the surrogate\n"
        " is at least as consistent with Monte Carlo as MC is with its own analytic\n"
        " benchmark - i.e. there is no evidence the surrogate learned 'analytic BS'\n"
        " instead of 'the MC operator'.)"
    )

    return rows


def run_xva_analysis(
    surrogate,
    market_data,
    fitted_params,
):

    print(
        "\nGenerating Monte Carlo paths..."
    )

    scheme = EulerMaruyama()

    mc_model = BlackScholesModel(
        scheme=scheme
    )

    # use the calibrated rate, not a hardcoded constant
    params = BlackScholesParams(
        r=fitted_params.r,
        sigma=fitted_params.sigma,
    )

    num_paths = 100
    num_steps = 252

    dt = 1.0 / 252.0

    mc_paths = scheme.generate_paths(
        s0=jnp.array([market_data.spot]),
        drift_fn=mc_model.drift,
        diffusion_fn=mc_model.diffusion,
        params=params,
        key=jax.random.PRNGKey(0),
        num_paths=num_paths,
        num_steps=num_steps,
        dt=dt,
    )

    mc_time_grid = jnp.linspace(
        0.0,
        1.0,
        num_steps + 1,
    )

    Visualizer.plot_mc_paths(
        mc_time_grid,
        mc_paths,
        num_paths=20,
    )

    S_path = mc_paths[:, :, 0]

    sigma = fitted_params.sigma
    r = fitted_params.r
    maturity = 1.0

    remaining_time = (
        maturity - mc_time_grid
    )

    remaining_time = jnp.maximum(
        remaining_time,
        1e-8,
    )

    def build_feature_paths(K: float):
        return jnp.stack(
            [
                S_path,

                K * jnp.ones_like(S_path),

                jnp.broadcast_to(
                    remaining_time,
                    S_path.shape,
                ),

                sigma * jnp.ones_like(S_path),

                r * jnp.ones_like(S_path),
            ],
            axis=-1,
        )

    print(
        "\nMC paths shape:",
        mc_paths.shape
    )

    # a deep ITM call is near-linear in S, so a single fixed K would hide
    # Greeks errors that ATM catches
    risk_engine = RiskEngine()

    K_atm = float(market_data.spot)
    strikes_to_validate = {
        "ITM": 0.85 * K_atm,
        "ATM": K_atm,
        "OTM": 1.15 * K_atm,
    }

    print("\n===== XVA VALIDATION (across moneyness) =====\n")

    results = {}

    for label, K in strikes_to_validate.items():

        feature_paths = build_feature_paths(K)

        xva_k = risk_engine.compute_xva_risk(
            surrogate,
            feature_paths,
            mc_time_grid,
            params.r,
        )

        xva_reference_k = risk_engine.compute_xva_risk_reference(
            feature_paths,
            mc_time_grid,
            params.r,
        )

        cva_error_k = (
            abs(xva_k["CVA"] - xva_reference_k["CVA"])
            / (abs(xva_reference_k["CVA"]) + 1e-12)
        )

        netxva_error_k = (
            abs(xva_k["NetXVA"] - xva_reference_k["NetXVA"])
            / (abs(xva_reference_k["NetXVA"]) + 1e-12)
        )

        print(
            f"{label:4s} (K={K:8.2f}) | "
            f"CVA Surrogate: {xva_k['CVA']:.6f} | "
            f"CVA Reference: {xva_reference_k['CVA']:.6f} | "
            f"CVA Error: {100*cva_error_k:.2f}% | "
            f"NetXVA Error: {100*netxva_error_k:.2f}%"
        )

        results[label] = {
            "K": K,
            "xva": xva_k,
            "xva_reference": xva_reference_k,
            "cva_error": cva_error_k,
            "netxva_error": netxva_error_k,
        }

    mc_reference_check(surrogate, market_data, fitted_params)

    # ATM is the primary, returned XVA (most curvature-sensitive)
    atm = results["ATM"]
    xva = atm["xva"]
    xva_reference = atm["xva_reference"]

    Visualizer.report_risk_metrics(
        xva
    )

    print("\n===== XVA VALIDATION (ATM, primary) =====\n")

    print(
        f"CVA Reference : "
        f"{xva_reference['CVA']:.6f}"
    )

    print(
        f"CVA Surrogate : "
        f"{xva['CVA']:.6f}"
    )

    print(
        f"DVA Reference : "
        f"{xva_reference['DVA']:.6f}"
    )

    print(
        f"DVA Surrogate : "
        f"{xva['DVA']:.6f}"
    )

    print(
        f"NetXVA Reference : "
        f"{xva_reference['NetXVA']:.6f}"
    )

    print(
        f"NetXVA Surrogate : "
        f"{xva['NetXVA']:.6f}"
    )

    print(
        f"CVA Error : "
        f"{100*atm['cva_error']:.2f}%"
    )

    print(
        f"NetXVA Error : "
        f"{100*atm['netxva_error']:.2f}%"
    )

    Visualizer.plot_exposure_profiles(
        mc_time_grid,
        xva["EPE"],
        xva["ENE"],
    )

    return xva
