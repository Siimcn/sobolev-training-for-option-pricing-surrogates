from typing import Optional, Tuple

from kalibrierung.market_data import MarketData
from kalibrierung.market_data_loader import MarketDataLoader

from surrogate_modeling.pricing_problem import CalibrationResult, calibrate_problem

from pipeline.config import ExperimentConfig


def load_and_calibrate(
    config: ExperimentConfig,
) -> Optional[Tuple[MarketData, CalibrationResult]]:
    """
    Load the option chain and fit the configured model to it.

    Which parameters are fitted, with which pricer, and what has to be
    assumed instead is the problem's business - this stage only decides
    that calibration happens and reports failure. Returns None if either
    step fails; the caller must stop.
    """

    try:
        market_data = _load_market_data(config)

        market_data.summary()

        print(
            "\n===== Calibration =====\n"
        )

        calibration = calibrate_problem(
            config.data.pricing_model,
            config=config,
            market_data=market_data,
        )

        print(
            f"Calibrated sigma: "
            f"{calibration.params.sigma:.4f}"
        )

        print(
            f"Calibrated rate : "
            f"{calibration.params.r:.4f}"
        )

    except Exception as e:
        print("\nMarket data unavailable:")
        print(e)
        return None

    return market_data, calibration


def _load_market_data(config: ExperimentConfig) -> MarketData:
    spot, strikes, maturities, prices, is_call = (
        MarketDataLoader.fetch_yahoo_options(
            ticker_symbol=config.market.ticker,
            max_maturities=config.market.max_maturities,
            cache_path=config.market.cache_path,
            use_cache=config.market.use_cache,
        )
    )

    return MarketData(
        spot=spot,
        strikes=strikes,
        maturities=maturities,
        market_prices=prices,
        is_call=is_call,
    )
