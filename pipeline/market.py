from typing import Optional, Tuple

from kalibrierung.calibrator import Calibrator
from kalibrierung.market_data import MarketData
from kalibrierung.market_data_loader import MarketDataLoader

from marktsimulation.black_scholes import black_scholes_price
from marktsimulation.pricing_model import BlackScholesParams

from pipeline.config import ExperimentConfig


def load_and_calibrate(
    config: ExperimentConfig,
) -> Optional[Tuple[MarketData, BlackScholesParams]]:
    """Returns None if loading or calibration fails; the caller must stop."""

    try:
        market_data = _load_market_data(config)

        market_data.summary()

        print(
            "\n===== Calibration =====\n"
        )

        fitted_params = _calibrate(market_data, config)

        print(
            f"Calibrated sigma: "
            f"{fitted_params.sigma:.4f}"
        )

    except Exception as e:
        print("\nMarket data unavailable:")
        print(e)
        return None

    return market_data, fitted_params


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


def _calibrate(
    market_data: MarketData,
    config: ExperimentConfig,
) -> BlackScholesParams:

    def calibration_pricer(params, strikes, maturities, is_call):
        return black_scholes_price(
            params=params,
            strikes=strikes,
            maturities=maturities,
            is_call=is_call,
            spot=market_data.spot,
        )

    calibrator = Calibrator(
        pricing_fn=calibration_pricer
    )

    initial_params = BlackScholesParams(
        r=config.market.initial_rate,
        sigma=config.market.initial_sigma,
    )

    fitted_params, _ = calibrator.calibrate(
        initial_params,
        market_data,
    )

    return fitted_params
