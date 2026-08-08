from typing import Dict, Optional

from kalibrierung.market_data import MarketData

from marktsimulation.pricing_model import BlackScholesParams

from risk_visualisierung.xva_analysis import run_xva_analysis

from surrogate_modeling.surrogate_model import SurrogateModel

from pipeline.config import ExperimentConfig


def run_risk_analysis(
    surrogate: SurrogateModel,
    market_data: MarketData,
    fitted_params: BlackScholesParams,
    config: ExperimentConfig,
) -> Optional[Dict[str, float]]:
    """
    Returns None when XVA cannot be run for the configured pricing model.

    The exposure simulation evolves a single spot and its reference prices
    with the closed-form Black-Scholes formula. A basket surrogate expects
    [S_1, ..., S_n, K, T] instead, and a basket has no closed form - and
    since both feature vectors are the same length for n = 3, feeding one
    to the other would silently produce nonsense rather than fail.
    """

    if config.is_basket:
        print(
            "\nSkipping XVA: the exposure simulation and its analytic "
            "reference are single-asset only."
        )
        return None

    return run_xva_analysis(surrogate, market_data, fitted_params)
