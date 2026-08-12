"""
Every pricing problem the surrogate can be trained on.

Importing this package registers all of them, so `available_problems()` and
`data.pricing_model` see the full set. One module per model family; adding a
model means adding a module here and importing it below.

See `docs/code_structure/adding_a_pricing_model.md` for the full walkthrough.
"""

from surrogate_modeling.problems.bachelier import (
    BACHELIER,
    BASKET_BACHELIER,
    BachelierProblem,
    BasketBachelierProblem,
    calibrate_bachelier,
    calibrate_basket_bachelier,
)
from surrogate_modeling.problems.black_scholes import (
    BASKET_BLACK_SCHOLES,
    BLACK_SCHOLES,
    BasketBlackScholesProblem,
    BlackScholesProblem,
    calibrate_basket,
    calibrate_black_scholes,
)
from surrogate_modeling.problems.heston import (
    BASKET_HESTON,
    HESTON,
    BasketHestonProblem,
    HestonProblem,
    calibrate_basket_heston,
    calibrate_heston,
)

__all__ = [
    "BACHELIER",
    "BASKET_BACHELIER",
    "BASKET_BLACK_SCHOLES",
    "BASKET_HESTON",
    "BLACK_SCHOLES",
    "HESTON",
    "BachelierProblem",
    "BasketBachelierProblem",
    "BasketBlackScholesProblem",
    "BasketHestonProblem",
    "BlackScholesProblem",
    "HestonProblem",
    "calibrate_bachelier",
    "calibrate_basket",
    "calibrate_basket_bachelier",
    "calibrate_basket_heston",
    "calibrate_black_scholes",
    "calibrate_heston",
]
