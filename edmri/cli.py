import argparse

from .optimise import optimise_experiment
from .inference import apply_trained_model


def optimise_cli():
    """Command-line interface for experiment optimisation."""

    parser = argparse.ArgumentParser(
        description="Optimise an MRI acquisition protocol using TADRED."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input MRI signals (.npy or NIfTI).",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target parameter maps or values.",
    )

    parser.add_argument(
        "--superdesign",
        required=True,
        help="Full acquisition protocol (.txt).",
    )

    parser.add_argument(
        "--protocol-size",
        type=float,
        default=0.5,
        help="Fraction of measurements to retain (default: 0.5).",
    )

    parser.add_argument(
        "--mask",
        default=None,
        help="Optional NIfTI mask.",
    )

    parser.add_argument(
        "--output-dir",
        default="edmri_output",
        help="Directory for results.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of TADRED optimisation stages.",
    )

    args = parser.parse_args()

    optimise_experiment(
        input_data=args.input,
        target_data=args.target,
        superdesign=args.superdesign,
        opt_protocol_size=args.protocol_size,
        mask=args.mask,
        output_dir=args.output_dir,
        n_iterations_tadred=args.iterations,
    )


def apply_cli():
    """Command-line interface for applying a trained model."""

    parser = argparse.ArgumentParser(
        description="Apply a trained TADRED model to MRI data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input MRI data (.npy or NIfTI).",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )

    parser.add_argument(
        "--mask",
        default=None,
        help="Optional NIfTI mask.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output file (.npy or .nii.gz).",
    )

    args = parser.parse_args()

    apply_trained_model(
        input_data=args.input,
        trained_model=args.model,
        mask=args.mask,
        output_file=args.output,
    )