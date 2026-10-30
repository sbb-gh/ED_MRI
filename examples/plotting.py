"""Plot SANDI simulation data, TADRED optimisation results, and predictions.

Run from the ED_MRI repository root:

    python examples/plotting.py \
        --data-dir examples/data \
        --optimisation-dir results/optimisation_results \
        --application-dir results/application_results

The script searches these directories recursively for the required files.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PARAM_NAMES = [
    r"$f_{neurite}$",
    r"$f_{soma}$",
    r"$D_{neurite}$ ($\mu$m$^2$/ms)",
    r"$D_{extra}$ ($\mu$m$^2$/ms)",
    r"$R_{soma}$ ($\mu$m)",
]


def find_file(directory: Path, filename: str) -> Path:
    """Find a uniquely named file, allowing TADRED's nested output folders."""
    matches = sorted(directory.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename!r} under {directory}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {filename!r} files found under {directory}:\n"
            + "\n".join(map(str, matches))
        )
    return matches[0]


def load_inputs(data_dir: Path, optimisation_dir: Path):
    signals = np.load(find_file(data_dir, "superdesign_signals.npy"))
    targets = np.load(find_file(data_dir, "superdesign_targets.npy"))

    protocol = np.loadtxt(
        find_file(data_dir, "superdesign_protocol.txt"),
        ndmin=2,
    )

    indices = np.loadtxt(
        find_file(optimisation_dir, "optimised_indices.txt"),
        dtype=int,
        ndmin=1,
    ).reshape(-1)

    if signals.ndim != 2 or targets.ndim != 2:
        raise ValueError("Signals and targets must be 2D arrays")

    if protocol.shape != (signals.shape[1], 3):
        raise ValueError(
            "Expected SANDI protocol with one (delta, Delta, G) row "
            f"per signal column; got {protocol.shape} and signals {signals.shape}"
        )

    if targets.shape != (signals.shape[0], 5):
        raise ValueError(
            f"Expected five SANDI targets per sample; got {targets.shape}"
        )

    if np.any(indices < 0) or np.any(indices >= signals.shape[1]):
        raise ValueError("TADRED indices are outside the superdesign")

    return signals, targets, protocol, indices


def load_predictions(application_dir: Path):
    predictions_file = find_file(application_dir, "predictions.npy")
    return np.load(predictions_file)


def b_values(protocol):
    """PGSE b-values in ms/um^2; protocol columns in ms, ms, mT/m."""
    gamma = 2 * np.pi * 42.57747892e6  # rad/s/T
    delta = protocol[:, 0] * 1e-3
    Delta = protocol[:, 1] * 1e-3
    G = protocol[:, 2] * 1e-3
    return (gamma * G * delta) ** 2 * (Delta - delta / 3) * 1e-9


def plot_protocol(protocol, indices, output_dir):
    b = b_values(protocol)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(protocol[:, 1], b, s=20, alpha=0.3, label="Full superdesign")
    ax.scatter(
        protocol[indices, 1], b[indices], s=60, marker="x", linewidths=1.8,
        label="TADRED selected", zorder=3,
    )
    ax.set(xlabel=r"Diffusion time $\Delta$ (ms)", ylabel=r"b-value (ms/$\mu$m$^2$)",
           title=f"SANDI acquisition selection ({len(indices)}/{len(protocol)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "acquisition_selection.png", dpi=200)
    plt.close(fig)


def plot_signals(signals, protocol, indices, output_dir, n_examples=4):
    b = b_values(protocol)
    order = np.argsort(b)
    n_examples = min(n_examples, len(signals))
    fig, axes = plt.subplots(n_examples, 1, figsize=(8, 2.8 * n_examples), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(b[order], signals[i, order], ".", alpha=0.45, label="Full superdesign")
        ax.plot(b[indices], signals[i, indices], "x",
                   label="TADRED selected", zorder=3)
        ax.set(xlabel=r"b-value (ms/$\mu$m$^2$)", ylabel="Signal", title=f"Sample {i + 1}")
        if i == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "example_signals.png", dpi=200)
    plt.close(fig)



def plot_predictions(targets, predictions, output_dir):
    predictions = np.asarray(predictions)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} must match targets {targets.shape}. "
            "Use predictions for the same samples in the same order."
        )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i >= targets.shape[1]:
            ax.axis("off")
            continue
        truth, pred = targets[:, i], predictions[:, i]
        rmse = np.sqrt(np.mean((pred - truth) ** 2))
        ss_tot = np.sum((truth - truth.mean()) ** 2)
        r2 = 1 - np.sum((pred - truth) ** 2) / ss_tot if ss_tot > 0 else np.nan
        ax.scatter(truth, pred, s=8, alpha=0.3)
        lo = min(truth.min(), pred.min())
        hi = max(truth.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set(xlabel=f"True {PARAM_NAMES[i]}", ylabel=f"Predicted {PARAM_NAMES[i]}",
               title=f"RMSE={rmse:.3g}, R²={r2:.3f}")
    fig.suptitle("TADRED SANDI parameter predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_accuracy.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("examples/data/sandi"),
        help="Directory containing the oversampled simulation data.",
    )

    parser.add_argument(
        "--optimisation-dir",
        type=Path,
        default=Path("results/optimisation"),
        help="Directory containing TADRED optimisation results.",
    )

    parser.add_argument(
        "--application-dir",
        type=Path,
        default=Path("results/application"),
        help="Directory containing task network predictions.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory for saving figures.",
    )

    parser.add_argument(
        "--n-examples",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load simulation data and optimisation results
    signals, targets, protocol, indices = load_inputs(
        args.data_dir,
        args.optimisation_dir,
    )

    # Plot protocol and signals
    plot_protocol(protocol, indices, args.output_dir)

    plot_signals(
        signals,
        protocol,
        indices,
        args.output_dir,
        args.n_examples,
    )

    # Load application results and plot prediction accuracy
    try:
        predictions = load_predictions(args.application_dir)
    except FileNotFoundError:
        print("No application predictions found; skipping accuracy plots.")
    else:
        plot_predictions(targets, predictions, args.output_dir)

    print(f"Saved plots to {args.output_dir}")
    print(
        f"Samples: {len(signals)}, "
        f"measurements: {signals.shape[1]}, "
        f"selected: {len(indices)}"
    )


if __name__ == "__main__":
    main()
