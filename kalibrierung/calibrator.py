import jax.numpy as jnp
import optimistix as optx

from typing import Callable, Optional, Any

from kalibrierung.market_data import MarketData


class Calibrator:

    def __init__(
        self,
        pricing_fn: Callable,
        transform_fn: Optional[Callable] = None,
        inv_transform_fn: Optional[Callable] = None,
    ):

        self.pricing_fn = pricing_fn

        self.transform_fn = transform_fn
        self.inv_transform_fn = inv_transform_fn

    def _residuals(
        self,
        params,
        args,
    ):

        market_data = args

        eval_params = params

        if self.transform_fn is not None:
            eval_params = self.transform_fn(
                params
            )

        model_prices = self.pricing_fn(
            eval_params,
            market_data.strikes,
            market_data.maturities,
            market_data.is_call,
        )

        return (
            model_prices
            - market_data.market_prices
        ) * market_data.weights

    def calibrate(
        self,
        init_params,
        market_data: MarketData,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 500,
    ):

        y0 = init_params

        if self.inv_transform_fn is not None:
            y0 = self.inv_transform_fn(
                init_params
            )

        solver = optx.LevenbergMarquardt(
            rtol=rtol,
            atol=atol,
        )

        solution = optx.least_squares(
            fn=self._residuals,
            solver=solver,
            y0=y0,
            args=market_data,
            max_steps=max_steps,
            throw=False,
        )

        fitted_params = solution.value

        if self.transform_fn is not None:
            fitted_params = self.transform_fn(
                fitted_params
            )

        return (
            fitted_params,
            solution,
        )

    def pricing_error(
        self,
        params,
        market_data: MarketData,
    ):

        model_prices = self.pricing_fn(
            params,
            market_data.strikes,
            market_data.maturities,
            market_data.is_call,
        )

        residuals = (
            model_prices
            - market_data.market_prices
        )

        rmse = jnp.sqrt(
            jnp.mean(
                residuals**2
            )
        )

        mae = jnp.mean(
            jnp.abs(
                residuals
            )
        )

        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
        }

    def report(
        self,
        params,
        market_data: MarketData,
    ):

        metrics = self.pricing_error(
            params,
            market_data,
        )

        print("\n")
        print("=" * 50)
        print("CALIBRATION REPORT")
        print("=" * 50)

        print(
            f"RMSE : "
            f"{metrics['RMSE']:.6e}"
        )

        print(
            f"MAE  : "
            f"{metrics['MAE']:.6e}"
        )

        print("=" * 50)