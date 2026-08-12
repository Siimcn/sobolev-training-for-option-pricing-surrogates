import jax
import jax.numpy as jnp


class MonteCarloPricer:

    def __init__(self, model, payoff, value_fn=None, payoff_on_path=False):
        self.model = model
        self.payoff = payoff
        self.value_fn = value_fn
        self.payoff_on_path = payoff_on_path

    def price(self, s0, params, maturity, num_paths, num_steps, key):

        dt = maturity / num_steps

        paths = self.model.scheme.generate_paths(
            s0=s0,
            drift_fn=self.model.drift,
            diffusion_fn=self.model.diffusion,
            params=params,
            key=key,
            num_paths=num_paths,
            num_steps=num_steps,
            dt=dt,
            corr=self.model.noise_correlation(params),
        )

        if self.payoff_on_path:
            if self.value_fn is not None:
                underlying = jax.vmap(jax.vmap(self.value_fn))(paths)
            else:
                underlying = paths[:, :, 0]

        else:
            terminal_values = paths[:, -1]

            if self.value_fn is not None:
                underlying = jax.vmap(self.value_fn)(terminal_values)
            else:
                underlying = terminal_values

        payoff_values = jax.vmap(self.payoff)(underlying)

        return jnp.mean(payoff_values)
