import jax.numpy as jnp
import equinox as eqx

from typing import Optional


class MarketData(eqx.Module):

    spot: float

    strikes: jnp.ndarray
    maturities: jnp.ndarray
    market_prices: jnp.ndarray

    is_call: jnp.ndarray

    weights: jnp.ndarray

    def __init__(
        self,
        spot: float,
        strikes: jnp.ndarray,
        maturities: jnp.ndarray,
        market_prices: jnp.ndarray,
        is_call: jnp.ndarray,
        weights: Optional[jnp.ndarray] = None,
    ):

        self.spot = float(spot)

        self.strikes = jnp.asarray(
            strikes,
            dtype=jnp.float64,
        )

        self.maturities = jnp.asarray(
            maturities,
            dtype=jnp.float64,
        )

        self.market_prices = jnp.asarray(
            market_prices,
            dtype=jnp.float64,
        )

        self.is_call = jnp.asarray(
            is_call,
            dtype=bool,
        )

        if weights is None:

            self.weights = jnp.ones_like(
                self.market_prices
            )

        else:

            self.weights = jnp.asarray(
                weights,
                dtype=jnp.float64,
            )

        self._validate()

    def _validate(self):

        n = len(self.market_prices)

        if len(self.strikes) != n:
            raise ValueError(
                "strikes and prices have different lengths."
            )

        if len(self.maturities) != n:
            raise ValueError(
                "maturities and prices have different lengths."
            )

        if len(self.is_call) != n:
            raise ValueError(
                "is_call and prices have different lengths."
            )

        if len(self.weights) != n:
            raise ValueError(
                "weights and prices have different lengths."
            )

    def __len__(self):

        return len(self.market_prices)

    @property
    def num_instruments(self):

        return len(self)

    @property
    def calls(self):

        mask = self.is_call

        return MarketData(
            self.spot,
            self.strikes[mask],
            self.maturities[mask],
            self.market_prices[mask],
            self.is_call[mask],
            self.weights[mask],
        )

    @property
    def puts(self):

        mask = ~self.is_call

        return MarketData(
            self.spot,
            self.strikes[mask],
            self.maturities[mask],
            self.market_prices[mask],
            self.is_call[mask],
            self.weights[mask],
        )

    def summary(self):

        print("\n===== Market Data =====\n")

        print(
            f"Spot                  : "
            f"{self.spot:.2f}"
        )

        print(
            f"Number of instruments : "
            f"{self.num_instruments}"
        )

        print(
            f"Calls                 : "
            f"{int(jnp.sum(self.is_call))}"
        )

        print(
            f"Puts                  : "
            f"{int(jnp.sum(~self.is_call))}"
        )

        print(
            f"Strike range          : "
            f"[{float(jnp.min(self.strikes)):.2f}, "
            f"{float(jnp.max(self.strikes)):.2f}]"
        )

        print(
            f"Maturity range        : "
            f"[{float(jnp.min(self.maturities)):.3f}, "
            f"{float(jnp.max(self.maturities)):.3f}]"
        )

        print(
            f"Price range           : "
            f"[{float(jnp.min(self.market_prices)):.4f}, "
            f"{float(jnp.max(self.market_prices)):.4f}]"
        )

    def calibration_tuple(self):

        return (
            self.strikes,
            self.maturities,
            self.market_prices,
            self.is_call,
            self.weights,
        )