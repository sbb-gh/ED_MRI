
"""Generate example simulated MRI data using ED_MRI models."""

import argparse
from pathlib import Path

import numpy as np

from edmri import simulation


MODELS = {
    "ADC": simulation.ADC,
    "T1INV": simulation.T1INV,
    "SANDI": simulation.SANDI,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulated MRI data for ED_MRI examples."
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Optional acquisition protocol file, e.g. TADRED's optimised_protocol.txt.",
    )
    
    parser.add_argument(
        "--model",
        type=str.upper,
        choices=MODELS.keys(),
        default="SANDI",
        help="Simulation model (default: SANDI).",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of simulated samples (default: 1000).",
    )

    parser.add_argument(
        "--snr-min",
        type=float,
        default=10,
        help="Minimum simulation SNR (default: 10).",
    )

    parser.add_argument(
        "--snr-max",
        type=float,
        default=50,
        help="Maximum simulation SNR (default: 50).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory for generated data.",
    )

    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("--n-samples must be positive.")

    if args.snr_min <= 0 or args.snr_max < args.snr_min:
        parser.error("Invalid SNR range.")

    np.random.seed(args.seed)

    # Initialise the selected simulation model.
    experiment = MODELS[args.model](initialise_classical=False)

    experiment.create_params(args.n_samples)

    if args.protocol is not None:
        acquisition_scheme = np.loadtxt(args.protocol, ndmin=2)

        # For ADC and T1INV, acquisition parameters are 1D.
        if args.model in ("ADC", "T1INV"):
            acquisition_scheme = acquisition_scheme.ravel()

    else:
        acquisition_scheme = experiment.acquisition_scheme_dense

    # Simulate signals using the selected acquisition scheme.
    signals = experiment.model_forward(
        acquisition_scheme,
        experiment.params_for_model,
    )

    # Add Rician noise.
    signals = experiment.add_noise(
        signals,
        snr_range=(args.snr_min, args.snr_max),
    ).astype(np.float32)

    targets = np.asarray(experiment.params_target).copy()
    superdesign = np.asarray(acquisition_scheme)

    # Validate output dimensions.
    if signals.shape[0] != targets.shape[0]:
        raise ValueError("Signal and target sample counts differ.")

    if signals.shape[1] != superdesign.shape[0]:
        raise ValueError(
            "Signal acquisition count does not match superdesign."
        )

    # Save model-specific files.
    
    protocol_name = "tadred" if args.protocol is not None else "superdesign"
    prefix = f"{protocol_name}"

    output_dir = args.output_dir / args.model.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{prefix}_signals.npy", signals)
    np.save(output_dir / f"{prefix}_targets.npy", targets)

    np.savetxt(
        output_dir / f"{prefix}_protocol.txt",
        superdesign,
        fmt="%.8g",
)
    print("Simulated MRI data generation complete.")
    print(f"Model:       {args.model}")
    print(f"Signals dim:     {signals.shape}")
    print(f"Targets dim:     {targets.shape}")
    print(f"Protocol dim: {superdesign.shape}")
    print(f"Saved to:    {output_dir}")
    if args.protocol is not None:
        print(f"Protocol file: {args.protocol}")
    else:
        print("Protocol: Default superdesign for " f"{args.model}" " model.")



if __name__ == "__main__":
    main()
