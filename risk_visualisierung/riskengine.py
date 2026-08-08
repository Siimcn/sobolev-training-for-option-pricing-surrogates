import jax
import jax.numpy as jnp

from typing import Dict


from marktsimulation.black_scholes import (
    black_scholes_price,
)

from marktsimulation.pricing_model import (
    BlackScholesParams,
)



class RiskEngine:

    def __init__(
        self,
        recovery_rate: float = 0.4,
        hazard_rate: float = 0.02,
    ):
        self.recovery_rate = recovery_rate
        self.hazard_rate = hazard_rate

    def compute_xva_risk(
        self,
        surrogate,
        paths: jnp.ndarray,
        time_grid: jnp.ndarray,
        discount_rate: float = 0.0,
    ) -> Dict:

        evaluate = jax.vmap(
            jax.vmap(
                surrogate.predict_price
            )
        )

        V = evaluate(paths)

        discount_factors = jnp.exp(
            -discount_rate * time_grid
        )

        discounted_V = (
            V * discount_factors
        )

        EPE = jnp.mean(
            jnp.maximum(
                discounted_V,
                0.0,
            ),
            axis=0,
        )

        ENE = jnp.mean(
            jnp.maximum(
                -discounted_V,
                0.0,
            ),
            axis=0,
        )

        survival = jnp.exp(
            -self.hazard_rate
            * time_grid
        )

        default_probs = (
            survival[:-1]
            - survival[1:]
        )

        # survival differences are one shorter than time_grid
        default_probs = jnp.concatenate(
            [
                default_probs,
                jnp.array([0.0]),
            ]
        )

        LGD = (
            1.0
            - self.recovery_rate
        )

        cva = LGD * jnp.sum(
            EPE * default_probs
        )

        dva = LGD * jnp.sum(
            ENE * default_probs
        )

        return {
            "EPE": EPE,
            "ENE": ENE,
            "CVA": float(cva),
            "DVA": float(dva),
            "NetXVA": float(cva - dva),
            "V_matrix": V,
        }
    
    
    def compute_xva_risk_reference(
        self,
        paths: jnp.ndarray,
        time_grid: jnp.ndarray,
        discount_rate: float = 0.0,
    ):

        evaluate = jax.vmap(
            jax.vmap(
                self.value_state
            )
        )

        V = evaluate(paths)

        discount_factors = jnp.exp(
            -discount_rate * time_grid
        )

        discounted_V = (
            V * discount_factors
        )

        EPE = jnp.mean(
            jnp.maximum(
                discounted_V,
                0.0,
            ),
            axis=0,
        )

        ENE = jnp.mean(
            jnp.maximum(
                -discounted_V,
                0.0,
            ),
            axis=0,
        )

        survival = jnp.exp(
            -self.hazard_rate
            * time_grid
        )

        default_probs = (
            survival[:-1]
            - survival[1:]
        )

        # survival differences are one shorter than time_grid
        default_probs = jnp.concatenate(
            [
                default_probs,
                jnp.array([0.0]),
            ]
        )

        LGD = (
            1.0
            - self.recovery_rate
        )

        cva = LGD * jnp.sum(
            EPE * default_probs
        )

        dva = LGD * jnp.sum(
            ENE * default_probs
        )

        return {
            "EPE": EPE,
            "ENE": ENE,
            "CVA": float(cva),
            "DVA": float(dva),
            "NetXVA": float(cva - dva),
            "V_matrix": V,
        }


    
    def value_state(self, x):

        maturity = jnp.maximum(
            x[2],
            1e-8,
        )

        params = BlackScholesParams(
            r=x[4],
            sigma=x[3],
        )

        return black_scholes_price(
            params=params,
            strikes=jnp.array([x[1]]),
            maturities=jnp.array([maturity]),
            is_call=jnp.array([True]),
            spot=x[0],
        )[0]



    def report(
        self,
        metrics: Dict,
    ):

        print("\n===== XVA REPORT =====")

        print(
            f"CVA     : {metrics['CVA']:.6e}"
        )

        print(
            f"DVA     : {metrics['DVA']:.6e}"
        )

        print(
            f"Net XVA : {metrics['NetXVA']:.6e}"
        )