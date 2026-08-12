from typing import Optional, Tuple

from calibration.market_data import MarketData
from calibration.market_data_loader import MarketDataLoader

from surrogate_modeling.pricing_problem import CalibrationResult, calibrate_problem

from pipeline.config import ExperimentConfig


def load_and_calibrate(
    config: ExperimentConfig,
) -> Optional[Tuple[MarketData, CalibrationResult]]:
    """Load the option chain and fit the configured model to it."""

    try:
        market_data = _load_market_data(config)

        market_data.summary()

        print("\n===== Calibration =====\n")

        calibration = calibrate_problem(
            config.data.pricing_model, config=config, market_data=market_data
        )

        _print_fitted(calibration)

    except Exception as e:
        print("\nMarket data unavailable:")
        print(e)
        return None

    return market_data, calibration


def _print_fitted(calibration: CalibrationResult) -> None:
    """Report whatever the model fitted, without knowing its parameter names."""

    params = calibration.params

    fields = params._asdict() if hasattr(params, "_asdict") else {"params": params}

    print("\nFitted parameters:")

    for name, value in fields.items():
        try:
            print(f"  {name:<8s}: {float(value):.6f}")
        except (TypeError, ValueError):
            print(f"  {name:<8s}: {value}")

    if calibration.assumptions:
        print("\nAssumed, not fitted:")

        for name, value in calibration.assumptions.items():
            print(f"  {name:<24s}: {value}")


def _load_market_data(config: ExperimentConfig) -> MarketData:
    spot, strikes, maturities, prices, is_call = MarketDataLoader.fetch_yahoo_options(
        ticker_symbol=config.market.ticker,
        max_maturities=config.market.max_maturities,
        cache_path=config.market.cache_path,
        use_cache=config.market.use_cache,
    )

    return MarketData(
        spot=spot,
        strikes=strikes,
        maturities=maturities,
        market_prices=prices,
        is_call=is_call,
    )
