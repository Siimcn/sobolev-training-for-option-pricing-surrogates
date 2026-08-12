import json
import os

import yfinance as yf
import jax.numpy as jnp

from datetime import datetime
from typing import Optional, Tuple


class MarketDataLoader:

    @staticmethod
    def fetch_yahoo_options(
        ticker_symbol: str,
        max_maturities: int = 3,
        min_volume: int = 0,
        min_open_interest: int = 100,
        cache_path: Optional[str] = None,
        use_cache: bool = False,
    ) -> Tuple[
        float,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:

        if cache_path and use_cache and os.path.exists(cache_path):
            return MarketDataLoader.load_cache(cache_path)

        data = MarketDataLoader._fetch_from_yahoo(
            ticker_symbol=ticker_symbol,
            max_maturities=max_maturities,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
        )

        if cache_path:
            MarketDataLoader.save_cache(cache_path, ticker_symbol, *data)

        return data

    @staticmethod
    def _fetch_from_yahoo(
        ticker_symbol: str,
        max_maturities: int = 3,
        min_volume: int = 0,
        min_open_interest: int = 100,
    ):

        print(
            f"Loading option data for "
            f"{ticker_symbol}..."
        )

        ticker = yf.Ticker(ticker_symbol)

        spot = float(
            ticker.history(
                period="1d"
            )["Close"].iloc[-1]
        )

        expirations = ticker.options

        if not expirations:
            raise ValueError(
                f"No option expiries found for "
                f"{ticker_symbol}."
            )

        strikes = []
        maturities = []
        prices = []
        is_calls = []

        n_total = 0
        n_quoted = 0
        n_liquid = 0

        today = datetime.today()

        for expiry_str in expirations[:max_maturities]:

            expiry = datetime.strptime(
                expiry_str,
                "%Y-%m-%d",
            )

            T = (
                expiry - today
            ).days / 365.25

            if T <= 0.0:
                continue

            chain = ticker.option_chain(
                expiry_str
            )

            for option_df, call_flag in [
                (chain.calls, 1.0),
                (chain.puts, 0.0),
            ]:

                bid = option_df["bid"].fillna(0.0)
                ask = option_df["ask"].fillna(0.0)
                volume = option_df["volume"].fillna(0.0)
                open_interest = option_df["openInterest"].fillna(0.0)

                has_quote = (
                    (bid > 0.0)
                    & (ask > 0.0)
                    & (ask >= bid)
                )

                liquid = open_interest >= min_open_interest

                if min_volume > 0:
                    liquid = liquid | (volume >= min_volume)

                keep = has_quote & liquid

                n_total += len(option_df)
                n_quoted += int(has_quote.sum())
                n_liquid += int(liquid.sum())

                valid = option_df[keep]

                mid_prices = (
                    (bid[keep] + ask[keep]) / 2.0
                )

                for (_, row), mid_price in zip(
                    valid.iterrows(), mid_prices
                ):

                    strikes.append(
                        float(row["strike"])
                    )

                    maturities.append(T)

                    prices.append(float(mid_price))

                    is_calls.append(
                        call_flag
                    )

        print(
            f"Loaded {len(prices)} liquid options "
            f"(of {n_total} contracts: {n_quoted} quoted, "
            f"{n_liquid} above open-interest floor)."
        )

        if len(prices) == 0:

            raise ValueError(
                f"No liquid options remained after filtering "
                f"({n_total} contracts seen, {n_quoted} with a two-sided "
                f"quote, {n_liquid} with openInterest >= "
                f"{min_open_interest}). Try lowering min_open_interest, or "
                f"fetch while the US market is open (09:30-16:00 ET)."
            )

        if len(prices) < 50:
            print(
                f"WARNING: only {len(prices)} options passed the filters. "
                f"Quotes are thin outside US market hours (09:30-16:00 ET); "
                f"consider fetching during the session and caching the "
                f"result via cache_path."
            )

        return (
            spot,
            jnp.asarray(
                strikes,
                dtype=jnp.float64,
            ),
            jnp.asarray(
                maturities,
                dtype=jnp.float64,
            ),
            jnp.asarray(
                prices,
                dtype=jnp.float64,
            ),
            jnp.asarray(
                is_calls,
                dtype=bool,
            ),
        )

    @staticmethod
    def save_cache(
        path: str,
        ticker_symbol: str,
        spot,
        strikes,
        maturities,
        prices,
        is_call,
    ) -> None:

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "ticker": ticker_symbol,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "spot": float(spot),
            "strikes": [float(v) for v in strikes],
            "maturities": [float(v) for v in maturities],
            "prices": [float(v) for v in prices],
            "is_call": [bool(v) for v in is_call],
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        print(
            f"Cached {len(payload['prices'])} options to {path}"
        )

    @staticmethod
    def load_cache(path: str):

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        print(
            f"Using cached option chain: {path}\n"
            f"  ticker     : {payload.get('ticker', 'unknown')}\n"
            f"  fetched_at : {payload.get('fetched_at', 'unknown')}\n"
            f"  options    : {len(payload['prices'])}\n"
            f"  (delete this file or pass use_cache=False to refetch)"
        )

        return (
            float(payload["spot"]),
            jnp.asarray(payload["strikes"], dtype=jnp.float64),
            jnp.asarray(payload["maturities"], dtype=jnp.float64),
            jnp.asarray(payload["prices"], dtype=jnp.float64),
            jnp.asarray(payload["is_call"], dtype=bool),
        )
