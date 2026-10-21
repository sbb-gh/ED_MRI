import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from tadred import inference

    
    
def apply_task_network(
    input_data,
    trained_task_network,
    *,
    mask=None,
    output_file=None,
):
    """
    Apply a trained EDMRI/TADRED task network to new MRI data.

    Parameters
    ----------
    input_data : np.ndarray or str or pathlib.Path
        Data acquired using the optimised protocol.

        Array input should have shape:
            (n_samples, n_acquisitions)

        NIfTI input should have shape:
            (X, Y, Z, n_acquisitions)

    trained_task_network : torch.nn.Module or str or pathlib.Path
        Trained TADRED model, or path to a saved checkpoint.

    mask : np.ndarray or str or pathlib.Path, optional
        Optional 3D mask. Only voxels where mask > 0 are processed.

    output_file : str or pathlib.Path, optional
        Path at which to save the resulting task output.

    Returns
    -------
    output : np.ndarray
        Predicted task output.
    """

    input_is_nifti = False
    input_img = None
    mask_array = None

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------

    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)

        if input_path.suffix == ".npy":
            input_array = np.load(input_path)

        elif input_path.suffix == ".nii" or input_path.name.endswith(".nii.gz"):
            input_is_nifti = True

            input_img = nib.load(input_path)
            input_array = input_img.get_fdata(dtype=np.float32)

        else:
            raise ValueError(
                f"Unsupported input format: {input_path}. "
                "Expected .npy, .nii, or .nii.gz."
            )

    else:
        input_array = np.asarray(input_data)

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    if input_is_nifti:

        if input_array.ndim != 4:
            raise ValueError(
                "NIfTI input must have shape "
                "(X, Y, Z, n_acquisitions)."
            )

        spatial_shape = input_array.shape[:3]
        n_acquisitions = input_array.shape[-1]

        if mask is not None:

            if isinstance(mask, (str, Path)):
                mask_img = nib.load(mask)
                mask_array = mask_img.get_fdata() > 0
            else:
                mask_array = np.asarray(mask) > 0

            if mask_array.shape != spatial_shape:
                raise ValueError(
                    f"Mask shape {mask_array.shape} does not match "
                    f"input shape {spatial_shape}."
                )

            model_input = input_array[mask_array]

        else:
            model_input = input_array.reshape(
                -1,
                n_acquisitions,
            )

    else:

        if input_array.ndim != 2:
            raise ValueError(
                "Array input must have shape "
                "(n_samples, n_acquisitions)."
            )

        model_input = input_array

    model_input = model_input.astype(np.float32)

    # ---------------------------------------------------------
    # TADRED inference
    # ---------------------------------------------------------

    # Live trained model
    predictions = inference.apply_trained_task_network(trained_task_network,model_input)

    # ---------------------------------------------------------
    # Reconstruct image
    # ---------------------------------------------------------

    if input_is_nifti:

        if predictions.ndim == 1:
            predictions = predictions[:, None]

        n_targets = predictions.shape[-1]

        output = np.zeros(
            (*spatial_shape, n_targets),
            dtype=np.float32,
        )

        if mask_array is not None:
            output[mask_array] = predictions

        else:
            output = predictions.reshape(
                *spatial_shape,
                n_targets,
            )

    else:
        output = predictions

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if output_file is not None:

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if input_is_nifti:
            output_img = nib.Nifti1Image(
                output,
                input_img.affine,
                input_img.header,
            )

            nib.save(
                output_img,
                output_file,
            )

        else:
            np.save(
                output_file,
                output,
            )

    return output



def main():
    """Command-line interface for applying a trained task network."""

    parser = argparse.ArgumentParser(
        description=(
            "Apply a trained TADRED task network to new MRI data acquired with the corresponding optimised protocol."
        )
    )

    parser.add_argument(
    "input_data",
    type=Path,
    help="Path to input data acquired with optimised protocol (.npy, .nii, or .nii.gz).",
    )

    parser.add_argument(
        "trained_task_network",
        type=Path,
        help="Path to the TADRED trained task network (.pt).",
    )

    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=(
            "Optional 3D NIfTI mask. Inference is performed only on voxels where mask > 0. "
            "Voxels where mask <= 0 are ignored."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("edmri_output"),
        help="Directory for inference outputs (default: edmri_output).",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/validation/test splitting (default: 42).",
    )

    args = parser.parse_args()

    apply_task_network(
        input_data=args.input_data,
        trained_task_network=args.trained_task_network,
        mask=args.mask,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()