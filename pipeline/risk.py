from typing import Dict, Optional

from risk_visualization.xva_analysis import run_xva_analysis

from surrogate_modeling.pricing_problem import PricingProblem
from surrogate_modeling.surrogate_model import SurrogateModel

from pipeline.config import ExperimentConfig


def run_risk_analysis(
    surrogate: SurrogateModel, problem: PricingProblem, config: ExperimentConfig
) -> Optional[Dict[str, float]]:
    """
    Returns None when the stage is switched off or the problem cannot simulate
    the future states an exposure profile needs.
    """

    if not config.risk.enabled:
        return None

    return run_xva_analysis(
        surrogate,
        problem,
        horizon=config.risk.horizon,
        num_paths=config.risk.num_paths,
        num_steps=config.risk.num_steps,
        seed=config.risk.seed,
        min_maturity=config.data.min_maturity,
    )
