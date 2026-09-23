import argparse
import datetime
from pathlib import Path

import numpy as np

from tadred import tadred_main, utils
from tadred import data_processing

from helpers import load_array

def optimise_experiment(
        input_data,
        target_data,
        superdesign,
        opt_protocol_size: float = 0.5,
        *,
        mask=None,
        output_dir=None,
        random_state: int = 42,
        n_iterations_tadred: int = 5,
        **kwargs,
):
    """
    Run TADRED to optimise an MRI acquisition protocol and train the
    corresponding task network.

    Parameters
    ----------
    input_data : np.ndarray or str or pathlib.Path
        Oversampled MRI data, or the path to a `.npy` file containing the data.
        Shape is (n_samples, n_acquisitions).

    target_data : np.ndarray or str or pathlib.Path
        Target task outputs, or the path to a `.npy` file containing the
        targets.
        Shape is (n_samples, n_targets).

    superdesign : np.ndarray or str or pathlib.Path
        Acquisition scheme describing the full oversampled experiment, or the
        path to a text file containing the acquisition scheme.
        Shape is (n_acquisitions, n_acquisition_parameters).

    opt_protocol_size : float, optional
        Fraction of the superdesign acquisitions to retain in the optimised
        protocol. Default is 0.5.

    output_dir : str or pathlib.Path, optional
        Directory in which to save optimisation outputs.

    random_state : int, optional
        Random seed used when splitting data into train, validation and test
        sets. Default is 42.

    n_iterations_tadred : int, optional
        Number of TADRED subset-reduction iterations. Default is 5.

    **kwargs
        Additional arguments controlling experiment optimisation and network
        training.

    Returns
    -------
    tadred_result
        Dictionary containing the complete TADRED optimisation results.
    """


    # Load input and target data.
    input_data = load_array(input_data, mask=mask)
    target_data = load_array(target_data, mask=mask)

    # Load the superdesign.
    if isinstance(superdesign, (str, Path)):
        superdesign = np.loadtxt(superdesign)

    superdesign = np.asarray(superdesign)

    # Check that the paired data are compatible.
    if input_data.shape[0] != target_data.shape[0]:
        raise ValueError(
            "input_data and target_data must contain the same number of samples."
        )

    if input_data.shape[-1] != len(superdesign):
        raise ValueError(
            "The final dimension of input_data must match the number of "
            "acquisitions in the superdesign."
        )

    # Set up output directory.
    if output_dir is None:
        output_dir = Path("edmri_output")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


    # Neural network hyperparameters of TADRED
    # TODO: move to config file
    tadred_args = utils.load_base_args()

    tadred_args.network.num_units_score = [1000, 1000]
    tadred_args.network.num_units_task = [1000, 1000]
    tadred_args.other_options.save_output = True

    # Base filename for saving the trained model and results
    tadred_args.output.out_base = output_dir

    tadred_args.output.proj_name = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    #organise the data into dictionaries for passing to TADRED
    data = data_processing.split_train_val_test(
        input_data,
        target_data,
        random_state=random_state,
    )

    #these are the key user-defined options

    n_vol_superdesign = superdesign.shape[0] #number of volumes in the oversampled superdesign    

    n_vol_opt_protocol = int(n_vol_superdesign * opt_protocol_size) #number of volumes in the desired optimized protocol

    # Set linearly decreasing subset sizes  
    feature_set_sizes = np.linspace(
        n_vol_superdesign,
        n_vol_opt_protocol,
        n_iterations_tadred,
        dtype=int,
    ).tolist()

    tadred_args.tadred_train_eval.feature_set_sizes_Ci = feature_set_sizes
    tadred_args.tadred_train_eval.feature_set_sizes_evaluated = feature_set_sizes

    tadred_result = tadred_main.run(tadred_args, data)

    #extract the optimised protocol

    # final subset index
    V_last = tadred_result["args"]["tadred_train_eval"]["feature_set_sizes_Ci"][-1] # V_{T} in paper

    # Index of chosen acquisition parameters
    acq_params_tadred_index = tadred_result[V_last]["measurements"]

    # Chosen acquisition parameters
    optimised_protocol = superdesign[acq_params_tadred_index]

    np.savetxt(Path(output_dir, "optimised_indices.txt"),acq_params_tadred_index,fmt="%d")
    
    np.savetxt(Path(output_dir, "optimised_protocol.txt"), optimised_protocol, fmt="%s")


    return tadred_result


def main():
    """Command-line interface for EDMRI experiment optimisation."""

    parser = argparse.ArgumentParser(
        description=(
            "Optimise an MRI acquisition protocol and train the corresponding "
            "task network using TADRED."
        )
    )

    parser.add_argument(
    "input_data",
    type=Path,
    help="Path to oversampled input data (.npy, .nii, or .nii.gz).",
    )

    parser.add_argument(
        "target_data",
        type=Path,
        help="Path to target data (.npy, .nii, or .nii.gz).",
    )

    parser.add_argument(
        "superdesign",
        type=Path,
        help="Path to the superdesign acquisition scheme (.txt).",
    )

    parser.add_argument(
        "--opt-protocol-size",
        type=float,
        default=0.5,
        help=(
            "Fraction of superdesign acquisitions to retain "
            "(default: 0.5)."
        ),
    )

    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=(
            "Optional 3D NIfTI mask. Only voxels where mask > 0 "
            "are used for training."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("edmri_output"),
        help="Directory for optimisation outputs (default: edmri_output).",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation/test splitting (default: 42).",
    )

    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of TADRED optimisation iterations (default: 5).",
    )

    args = parser.parse_args()

    optimise_experiment(
        input_data=args.input_data,
        target_data=args.target_data,
        superdesign=args.superdesign,
        opt_protocol_size=args.opt_protocol_size,
        mask=args.mask,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_iterations_tadred=args.n_iterations,
    )


if __name__ == "__main__":
    main()