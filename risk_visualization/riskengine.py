import jax
import jax.numpy as jnp

from typing import Callable, Dict


class RiskEngine:
    """Exposure and XVA from a valuation along simulated future states."""

    def __init__(self, recovery_rate: float = 0.4, hazard_rate: float = 0.02):
        self.recovery_rate = recovery_rate
        self.hazard_rate = hazard_rate

    def compute_xva_risk(
        self,
        surrogate,
        paths: jnp.ndarray,
        time_grid: jnp.ndarray,
        discount_rate: float = 0.0,
    ) -> Dict:

        return self.compute_xva_risk_reference(
            surrogate.predict_price, paths, time_grid, discount_rate
        )

    def compute_xva_risk_reference(
        self,
        value_fn: Callable[[jnp.ndarray], jnp.ndarray],
        paths: jnp.ndarray,
        time_grid: jnp.ndarray,
        discount_rate: float = 0.0,
    ) -> Dict:
        """
        `value_fn` prices one feature row. Passing it in is what makes the
        reference model-agnostic - it used to be the closed-form Black-Scholes
        price, which silently mispriced anything else.
        """

        evaluate = jax.vmap(jax.vmap(value_fn))

        return self._exposure_metrics(evaluate(paths), time_grid, discount_rate)

    def _exposure_metrics(
        self, V: jnp.ndarray, time_grid: jnp.ndarray, discount_rate: float
    ) -> Dict:

        discount_factors = jnp.exp(-discount_rate * time_grid)

        discounted_V = V * discount_factors

        EPE = jnp.mean(jnp.maximum(discounted_V, 0.0), axis=0)

        ENE = jnp.mean(jnp.maximum(-discounted_V, 0.0), axis=0)

        survival = jnp.exp(-self.hazard_rate * time_grid)

        default_probs = survival[:-1] - survival[1:]

        default_probs = jnp.concatenate([default_probs, jnp.array([0.0])])

        LGD = 1.0 - self.recovery_rate

        cva = LGD * jnp.sum(EPE * default_probs)

        dva = LGD * jnp.sum(ENE * default_probs)

        return {
            "EPE": EPE,
            "ENE": ENE,
            "CVA": float(cva),
            "DVA": float(dva),
            "NetXVA": float(cva - dva),
            "V_matrix": V,
        }

    def report(self, metrics: Dict):

        print("\n===== XVA REPORT =====")

        print(f"CVA     : {metrics['CVA']:.6e}")

        print(f"DVA     : {metrics['DVA']:.6e}")

        print(f"Net XVA : {metrics['NetXVA']:.6e}")
