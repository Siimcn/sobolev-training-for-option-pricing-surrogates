import jax
import jax.numpy as jnp
import numpy as np

from marktsimulation.pricing_model import (
    BlackScholesParams,
    BlackScholesModel,
)

from marktsimulation.timesteppingscheme import (
    EulerMaruyama,
)

from marktsimulation.payoff import (
    EuropeanCall,
    sigmoid_smooth,
)

from marktsimulation.monte_carlo_pricer import (
    MonteCarloPricer,
)

from marktsimulation.sobolev_labels import (
    create_sobolev_labels,
)


# number of MC paths per label (price, gradient and HVP all differentiate
# through the same simulation); 50k needed for a stable HVP estimate
MC_NUM_PATHS = 50_000
MC_NUM_STEPS = 50


def _payoff_smoothing_width(spot, maturity, sigma):
    """
    Width of the payoff-smoothing kernel, scaled to the terminal
    price's dispersion: a fixed dollar width is too narrow to catch
    enough paths for a stable HVP estimate once S_T's spread grows.
    """
    dispersion = spot * sigma * jnp.sqrt(jnp.maximum(maturity, 1e-6))
    return jnp.maximum(0.05 * dispersion, 1e-3)


def bs_mc_price(
    x: jnp.ndarray,
    key: jnp.ndarray = None,
):
    """
    x = [S, K, T, sigma, r]

    `key` defaults to a fixed PRNGKey(0) so every training label uses
    common random numbers. Pass a different key for an independent
    re-simulation.
    """

    if key is None:
        key = jax.random.PRNGKey(0)

    spot = x[0]
    strike = x[1]
    maturity = x[2]
    sigma = x[3]
    r = x[4]

    params = BlackScholesParams(
        r=r,
        sigma=sigma,
    )

    scheme = EulerMaruyama()

    model = BlackScholesModel(
        scheme=scheme
    )

    smooth_w = _payoff_smoothing_width(spot, maturity, sigma)

    payoff = EuropeanCall(
        strike=strike,
        smooth_w=smooth_w,
    )

    pricer = MonteCarloPricer(
        model=model,
        payoff=payoff,
    )

    price = pricer.price(
        s0=jnp.array([spot]),
        params=params,
        maturity=maturity,
        num_paths=MC_NUM_PATHS,
        num_steps=MC_NUM_STEPS,
        key=key,
    )

    discount = jnp.exp(
        -r * maturity
    )

    return discount * price


def make_mc_calibration_pricer(
    spot: float,
    maturities: jnp.ndarray,
    seed: int = 0,
    num_paths: int = MC_NUM_PATHS,
    num_steps: int = MC_NUM_STEPS,
):
    """
    Build a `pricing_fn` for the Calibrator that prices the whole
    instrument set by Monte Carlo instead of the analytic formula.

    Every European option sharing a maturity can be priced off the same
    terminal sample, so one residual evaluation costs one path bundle per
    *distinct maturity*, not one per instrument. The grouping is read off
    the market data here, before the solve starts, because it has to be a
    static structure inside the traced residual.

    The key is fixed and independent of the parameters (common random
    numbers). Re-drawing paths at every Levenberg-Marquardt step would
    make the residual discontinuous in (r, sigma) and its Jacobian
    meaningless.

    Note that the resulting prices carry the same Euler-discretization
    and payoff-smoothing bias as the training labels, so the calibrated
    parameters do too.
    """

    maturities = np.asarray(maturities)

    num_instruments = len(maturities)

    # (maturity, indices of the instruments that share it)
    groups = [
        (
            float(T),
            jnp.asarray(
                np.where(maturities == T)[0]
            ),
        )
        for T in np.unique(maturities)
    ]

    base_key = jax.random.PRNGKey(seed)

    scheme = EulerMaruyama()

    model = BlackScholesModel(
        scheme=scheme
    )

    def mc_pricing_fn(
        params,
        strikes,
        maturities,
        is_call,
    ):

        prices = jnp.zeros(
            num_instruments
        )

        for group_index, (T, indices) in enumerate(groups):

            key = jax.random.fold_in(
                base_key,
                group_index,
            )

            paths = scheme.generate_paths(
                s0=jnp.array([spot]),
                drift_fn=model.drift,
                diffusion_fn=model.diffusion,
                params=params,
                key=key,
                num_paths=num_paths,
                num_steps=num_steps,
                dt=T / num_steps,
                corr=model.noise_correlation(params),
            )

            terminal = paths[:, -1, 0]

            # same construction as EuropeanPayoff, but vectorized over
            # every strike at this maturity instead of one at a time
            omega = jnp.where(
                is_call[indices],
                1.0,
                -1.0,
            )

            intrinsic = (
                omega[None, :]
                * (
                    terminal[:, None]
                    - strikes[indices][None, :]
                )
            )

            payoffs = sigmoid_smooth(
                intrinsic,
                _payoff_smoothing_width(
                    spot,
                    T,
                    params.sigma,
                ),
            )

            discount = jnp.exp(
                -params.r * T
            )

            prices = prices.at[indices].set(
                discount
                * jnp.mean(
                    payoffs,
                    axis=0,
                )
            )

        return prices

    return mc_pricing_fn


def bs_mc_feature_price(
    x: jnp.ndarray,
):
    return bs_mc_price(x)

def bs_mc_feature_gradient(
    x: jnp.ndarray,
):
    return jax.grad(
        bs_mc_feature_price
    )(x)

def bs_mc_feature_hessian(
    x: jnp.ndarray,
):
    return jax.hessian(
        bs_mc_feature_price
    )(x)

def bs_mc_feature_hvp(x: jnp.ndarray, v: jnp.ndarray):
    """
    True HVP of the Monte Carlo price estimator, via forward-over-reverse
    autodiff (this does not build the full Hessian).
    """
    _, hvp = jax.jvp(jax.grad(bs_mc_feature_price), (x,), (v,))
    return hvp

# sequential dataset generation to avoid OOM errors
def create_bs_mc_dataset(
    X: jnp.ndarray,
    sobolev_order: int = 1,
):
    # value_and_grad so the MC simulation runs once per sample, not twice
    price_and_grad_fn = jax.jit(jax.value_and_grad(bs_mc_feature_price))

    if sobolev_order >= 2:
        hess_fn = jax.jit(bs_mc_feature_hessian)

    prices = []
    gradients = []
    hessians = []

    print(f"Generating data for {len(X)} samples sequentially to save memory...")

    for i in range(len(X)):
        x = X[i]

        price, grad = price_and_grad_fn(x)
        prices.append(price)
        gradients.append(grad)

        if sobolev_order >= 2:
            hessians.append(hess_fn(x))

    prices = jnp.stack(prices)
    gradients = jnp.stack(gradients)

    if sobolev_order >= 2:
        hessians = jnp.stack(hessians)
    else:
        hessians = None

    return (
        prices,
        gradients,
        hessians,
    )

def generate_training_paths(
    x: jnp.ndarray,
    num_paths: int = 100,
):

    spot = x[0]
    maturity = x[2]
    sigma = x[3]
    r = x[4]

    params = BlackScholesParams(
        r=r,
        sigma=sigma,
    )

    scheme = EulerMaruyama()

    model = BlackScholesModel(
        scheme=scheme
    )

    num_steps = MC_NUM_STEPS

    dt = maturity / num_steps

    paths = scheme.generate_paths(
        s0=jnp.array([spot]),
        drift_fn=model.drift,
        diffusion_fn=model.diffusion,
        params=params,
        key=jax.random.PRNGKey(0),
        num_paths=num_paths,
        num_steps=num_steps,
        dt=dt,
    )

    time_grid = jnp.linspace(
        0.0,
        maturity,
        num_steps + 1,
    )

    return time_grid, paths

def create_bs_mc_dataset_with_hvps(
    X: jnp.ndarray,
    V: jnp.ndarray,  # per-sample random unit probe vectors
    sobolev_order: int = 2,
):
    return create_sobolev_labels(
        bs_mc_feature_price,
        X,
        V,
        sobolev_order=sobolev_order,
        message=f"Generating data with HVPs for {len(X)} samples...",
    )
