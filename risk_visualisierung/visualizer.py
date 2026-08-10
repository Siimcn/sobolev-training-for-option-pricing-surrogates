import jax
import jax.numpy as jnp
import numpy as np
import os

import matplotlib

_SHOW_PLOTS = os.environ.get(
    "SWP_SHOW_PLOTS", "0"
).strip().lower() in ("1", "true", "yes", "on")

if not _SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from typing import Dict


class Visualizer:

    logger = None

    show_plots = _SHOW_PLOTS


    @staticmethod
    def _feature_label(labels, index: int) -> str:
        """Falls back to a positional name, so an unlabelled model still plots."""

        if index < len(labels):
            return str(labels[index])

        return f"Input {index}"


    @staticmethod
    def _save(
        filename: str,
    ):

        if Visualizer.logger is not None:

            save_path = Visualizer.logger.path(
                filename
            )

        else:

            save_path = os.path.join(
                "results",
                filename,
            )

            os.makedirs(
                "results",
                exist_ok=True,
            )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved: {save_path}"
        )

        if Visualizer.show_plots:
            plt.show()
        plt.close()


    @staticmethod
    def set_logger(
        logger,
    ):
        Visualizer.logger = logger

    @staticmethod
    def set_show(
        show: bool,
    ):
        """Turn on-screen display on/off at runtime."""
        Visualizer.show_plots = bool(show)


    @staticmethod
    def _rolling_average(values, window: int = 10):
        """Trailing rolling average over the noisy per-epoch curve."""

        values = np.asarray(values, dtype=float)

        if len(values) < 3:
            return values

        window = max(1, min(window, len(values)))
        kernel = np.ones(window) / window

        return np.convolve(values, kernel, mode="valid")

    @staticmethod
    def _plot_log_series(
        ax,
        epochs,
        train_values,
        valid_values=None,
        label: str = "Loss",
        smooth_window: int = 10,
        floor: float = 1e-12,
    ):
        """
        One train/valid pair on a log-scale y-axis, raw curve faint and
        smoothed curve solid. Log scale because the loss drops several
        orders of magnitude in the first ~10-20 epochs.
        """

        train_values = np.asarray(train_values, dtype=float)
        n_train = len(train_values)
        train_epochs = np.asarray(epochs[:n_train])

        if n_train > 0:
            train_floor = np.maximum(train_values, floor)
            ax.plot(train_epochs, train_floor, color="tab:blue", alpha=0.25, linewidth=1)

            smoothed = Visualizer._rolling_average(train_floor, smooth_window)
            smoothed_epochs = train_epochs[n_train - len(smoothed):]
            ax.plot(smoothed_epochs, smoothed, color="tab:blue", linewidth=2, label=f"Train {label}")

        if valid_values is not None and len(valid_values) > 0:
            valid_values = np.asarray(valid_values, dtype=float)
            n_valid = len(valid_values)
            valid_epochs = np.asarray(epochs[:n_valid])

            valid_floor = np.maximum(valid_values, floor)
            ax.plot(valid_epochs, valid_floor, color="tab:orange", alpha=0.25, linewidth=1)

            smoothed_v = Visualizer._rolling_average(valid_floor, smooth_window)
            smoothed_v_epochs = valid_epochs[n_valid - len(smoothed_v):]
            ax.plot(smoothed_v_epochs, smoothed_v, color="tab:orange", linewidth=2, label=f"Valid {label}")

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    @staticmethod
    def plot_training_history(
        history: Dict,
        filename: str = "training_history.png",
        zoom_start_epoch: int = 10,
        smooth_window: int = 10,
    ):
        """
        Training diagnostics figure (2x3 grid).

        The loss components get their own panels because the total loss
        is dominated by whichever term has the largest magnitude, so it
        can look healthy while gradient/HVP is barely learning (or
        overfitting) underneath.
        """

        train_loss = history.get("train_loss", [])
        valid_loss = history.get("valid_loss", [])
        epochs = np.arange(len(train_loss))

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        Visualizer._plot_log_series(
            axes[0, 0], epochs, train_loss, valid_loss,
            label="Total Sobolev Loss", smooth_window=smooth_window,
        )
        axes[0, 0].set_title("Total Loss - full range (log scale)")

        zoom_start = min(zoom_start_epoch, max(len(train_loss) - 1, 0))
        z_epochs = epochs[zoom_start:]
        z_train = np.asarray(train_loss[zoom_start:], dtype=float)
        z_valid = (
            np.asarray(valid_loss[zoom_start:], dtype=float)
            if len(valid_loss) > zoom_start
            else np.array([])
        )

        ax = axes[0, 1]
        if len(z_train) > 0:
            ax.plot(z_epochs, z_train, color="tab:blue", alpha=0.25, linewidth=1)
            sm = Visualizer._rolling_average(z_train, smooth_window)
            ax.plot(z_epochs[len(z_epochs) - len(sm):], sm, color="tab:blue", linewidth=2, label="Train (smoothed)")
        if len(z_valid) > 0:
            ax.plot(z_epochs[:len(z_valid)], z_valid, color="tab:orange", alpha=0.25, linewidth=1)
            smv = Visualizer._rolling_average(z_valid, smooth_window)
            ax.plot(
                z_epochs[:len(z_valid)][len(z_valid) - len(smv):], smv,
                color="tab:orange", linewidth=2, label="Valid (smoothed)",
            )
        ax.set_title(f"Total Loss - zoomed to epoch >= {zoom_start} (linear scale)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Sobolev Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        Visualizer._plot_log_series(
            axes[0, 2], epochs, history.get("train_price_rmse", []), history.get("valid_price_rmse", []),
            label="Price RMSE", smooth_window=smooth_window,
        )
        axes[0, 2].set_title("Price RMSE (log scale)")

        component_panels = [
            ("train_price_loss", "valid_price_loss", "Price Loss"),
            ("train_gradient_loss", "valid_gradient_loss", "Gradient Loss"),
            ("train_hessian_loss", "valid_hessian_loss", "HVP / Hessian Loss"),
        ]

        for ax, (train_key, valid_key, label) in zip(axes[1, :], component_panels):

            train_vals = history.get(train_key, [])

            # a term excluded from this run is all zeros - flag it instead
            # of plotting a misleading flat line at the log-axis floor
            is_unused = len(train_vals) == 0 or all(
                float(v) == 0.0 for v in train_vals
            )

            if is_unused:
                ax.text(
                    0.5, 0.5,
                    f"{label}\nnot part of the training objective\nin this run (sobolev_order too low)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray",
                )
                ax.set_title(f"{label} (not used)")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("lightgray")
            else:
                Visualizer._plot_log_series(
                    ax, epochs, train_vals, history.get(valid_key, []),
                    label=label, smooth_window=smooth_window,
                )
                ax.set_title(f"{label} (log scale, train vs. valid)")

        fig.suptitle("Sobolev Training Diagnostics", fontsize=14)
        plt.tight_layout()

        Visualizer._save(
            filename
        )


    @staticmethod
    def plot_mc_paths(
        time_grid,
        paths,
        num_paths=20,
        filename: str = "mc_paths.png"
    ):

        plt.figure(figsize=(10, 6))

        sample_paths = paths[:num_paths]

        if sample_paths.ndim == 3:
            sample_paths = sample_paths[:, :, 0]

        plt.plot(
            np.asarray(time_grid),
            np.asarray(sample_paths.T),
            alpha=0.6,
        )

        plt.xlabel("Time")
        plt.ylabel("Asset Price")
        plt.title("Monte Carlo Paths")

        plt.grid(True)

        plt.tight_layout()
        Visualizer._save(filename)

    @staticmethod
    def plot_price_comparison(
        true_prices: jnp.ndarray,
        predicted_prices: jnp.ndarray,
        filename: str = "price_comparison.png",
    ):

        true_prices = np.asarray(true_prices)
        predicted_prices = np.asarray(
            predicted_prices
        )

        plt.figure(figsize=(7, 7))

        plt.scatter(
            true_prices,
            predicted_prices,
            alpha=0.4,
        )

        lo = min(
            true_prices.min(),
            predicted_prices.min(),
        )

        hi = max(
            true_prices.max(),
            predicted_prices.max(),
        )

        plt.plot(
            [lo, hi],
            [lo, hi],
            "r--",
            linewidth=2,
        )

        plt.xlabel("True Prices")
        plt.ylabel("Predicted Prices")

        plt.title(
            "Surrogate vs Ground Truth"
        )

        plt.grid(True)

        plt.tight_layout()
        Visualizer._save(filename)

    @staticmethod
    def plot_exposure_profiles(
        time_grid: jnp.ndarray,
        epe: jnp.ndarray,
        ene: jnp.ndarray,
        filename: str = "exposure_profiles.png"
    ):

        plt.figure(figsize=(10, 5))

        plt.plot(
            time_grid,
            epe,
            label="EPE",
            linewidth=2,
        )

        plt.plot(
            time_grid,
            ene,
            label="ENE",
            linewidth=2,
        )

        plt.fill_between(
            np.asarray(time_grid),
            np.asarray(epe),
            alpha=0.2,
        )

        plt.fill_between(
            np.asarray(time_grid),
            np.asarray(ene),
            alpha=0.2,
        )

        plt.xlabel("Time")
        plt.ylabel("Exposure")

        plt.title("Exposure Profile")

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        Visualizer._save(filename)

    @staticmethod
    def plot_surrogate_surface(
        surrogate,
        fixed_input: jnp.ndarray,
        x_idx: int,
        y_idx: int,
        x_range,
        y_range,
        feature_labels=(),
        grid_points: int = 50,
        filename: str = "surrogate_surface.png",
    ):
        """
        A 2-D slice through the surrogate's input space.

        `feature_labels` names the axes. It is passed in rather than known
        here, because which features exist depends on what is being priced
        - assuming [S, K, T, sigma, r] mislabels every other model.
        """

        x = jnp.linspace(
            x_range[0],
            x_range[1],
            grid_points,
        )

        y = jnp.linspace(
            y_range[0],
            y_range[1],
            grid_points,
        )

        X, Y = jnp.meshgrid(x, y)

        X_flat = X.reshape(-1)
        Y_flat = Y.reshape(-1)

        inputs = jnp.tile(
            fixed_input,
            (len(X_flat), 1),
        )

        inputs = inputs.at[:, x_idx].set(
            X_flat
        )

        inputs = inputs.at[:, y_idx].set(
            Y_flat
        )

        Z = jax.vmap(
            surrogate.predict_price
        )(inputs)

        Z = Z.reshape(
            grid_points,
            grid_points,
        )

        fig = plt.figure(
            figsize=(10, 7)
        )

        ax = fig.add_subplot(
            111,
            projection="3d",
        )

        ax.plot_surface(
            np.asarray(X),
            np.asarray(Y),
            np.asarray(Z),
            cmap="viridis",
        )

        x_label = Visualizer._feature_label(feature_labels, x_idx)
        y_label = Visualizer._feature_label(feature_labels, y_idx)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_zlabel("Price")

        ax.set_title(
            f"Option Price Surface: {x_label} vs {y_label}"
        )

        plt.tight_layout()
        Visualizer._save(filename)

    @staticmethod
    def report_risk_metrics(
        metrics: Dict,
    ):

        print("\n")
        print("=" * 50)
        print("RISK REPORT")
        print("=" * 50)

        print(
            f"CVA     : {metrics['CVA']:.6e}"
        )

        print(
            f"DVA     : {metrics['DVA']:.6e}"
        )

        print(
            f"Net XVA : {metrics['NetXVA']:.6e}"
        )

        print("=" * 50)

    @staticmethod
    def report_metrics(
        metrics: Dict,
    ):

        print("\n")
        print("=" * 50)
        print("SURROGATE EVALUATION")
        print("=" * 50)

        for key, value in metrics.items():

            if isinstance(
                value,
                (float, np.floating),
            ):
                print(
                    f"{key:30s}: "
                    f"{value:.6e}"
                )

        print("=" * 50)