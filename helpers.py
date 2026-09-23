from pathlib import Path

import nibabel as nib
import numpy as np


def load_array(data, mask=None):
    """
    Load data from a NumPy array, .npy file, or NIfTI file.

    For NIfTI data, spatial dimensions are flattened so that the returned
    array has shape (n_samples, n_features), with the final NIfTI dimension
    treated as the feature dimension.

    If a mask is provided, only voxels where mask > 0 are returned.

    Parameters
    ----------
    data : np.ndarray or str or pathlib.Path
        Input array or path to .npy, .nii, or .nii.gz file.

    mask : np.ndarray or str or pathlib.Path, optional
        3D binary mask. Only voxels where mask > 0 are retained.

    Returns
    -------
    np.ndarray
        Array with shape (n_samples, n_features).
    """
    if isinstance(data, np.ndarray):
        array = data

    else:
        data = Path(data)

        if data.suffix == ".npy":
            array = np.load(data)

        elif data.suffix == ".nii" or data.name.endswith(".nii.gz"):
            img = nib.load(data)
            array = img.get_fdata(dtype=np.float32)

        else:
            raise ValueError(
                f"Unsupported data format: {data}. "
                "Expected .npy, .nii, or .nii.gz."
            )

    # Already in (samples, features) format
    if array.ndim == 2:
        if mask is not None:
            raise ValueError(
                "A spatial mask cannot be applied to 2D array data."
            )
        return array

    # Treat a 3D image as having one feature per voxel
    if array.ndim == 3:
        array = array[..., np.newaxis]

    if array.ndim != 4:
        raise ValueError(
            f"Expected 2D array or 3D/4D image data, "
            f"but received shape {array.shape}."
        )

    # Apply mask
    if mask is not None:
        if isinstance(mask, (str, Path)):
            mask = nib.load(mask).get_fdata()

        mask = np.asarray(mask)

        if mask.ndim != 3:
            raise ValueError(
                f"Mask must be 3D, but received shape {mask.shape}."
            )

        if mask.shape != array.shape[:3]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"image shape {array.shape[:3]}."
            )

        return array[mask > 0]

    # No mask: flatten all spatial dimensions
    return array.reshape(-1, array.shape[-1])






def extract_optimised_measurements(
    input_data,
    optimised_indices,
    *,
    output_file=None,
):
    """
    Extract the optimised measurements from a full MRI dataset.

    Parameters
    ----------
    input_data : np.ndarray or str or pathlib.Path
        Full MRI dataset.

        Array input should have shape:
            (..., n_acquisitions)

        NIfTI input should have shape:
            (X, Y, Z, n_acquisitions)

        The final dimension is assumed to correspond to acquisitions.

    optimised_indices : np.ndarray or str or pathlib.Path
        Indices of the selected acquisitions, or the path to a text file
        containing those indices.

    output_file : str or pathlib.Path, optional
        Optional path for saving the subsetted dataset.

        If `input_data` is a NIfTI file, the output is saved as NIfTI.
        Otherwise, the output is saved as a NumPy `.npy` file.

    Returns
    -------
    output : np.ndarray
        Dataset containing only the optimised measurements.

    indices : np.ndarray
        Indices of the selected measurements in the full dataset.
    """

    # Load optimised indices if a filename was provided.
    if isinstance(optimised_indices, (str, Path)):
        optimised_indices = np.loadtxt(
            optimised_indices,
            dtype=int,
        )

    indices = np.asarray(
        optimised_indices,
        dtype=int,
    )

    # np.loadtxt returns a scalar if there is only one index.
    indices = np.atleast_1d(indices)

    if indices.ndim != 1:
        raise ValueError(
            "optimised_indices must be a one-dimensional array of indices."
        )

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------

    input_is_nifti = False
    input_img = None

    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)

        if input_path.suffix == ".npy":
            data = np.load(input_path)

        elif input_path.suffix == ".nii" or input_path.name.endswith(".nii.gz"):
            input_is_nifti = True

            input_img = nib.load(input_path)
            data = input_img.get_fdata(dtype=np.float32)

        else:
            raise ValueError(
                f"Unsupported input format: {input_path}. "
                "Expected .npy, .nii, or .nii.gz."
            )

    else:
        data = np.asarray(input_data)

    # ---------------------------------------------------------
    # Validate indices
    # ---------------------------------------------------------

    n_acquisitions = data.shape[-1]

    if np.any(indices < 0):
        raise ValueError(
            "optimised_indices contains negative indices."
        )

    if np.any(indices >= n_acquisitions):
        raise ValueError(
            "optimised_indices contains an index larger than the number "
            f"of acquisitions in the input data ({n_acquisitions})."
        )

    # ---------------------------------------------------------
    # Extract optimised measurements
    # ---------------------------------------------------------

    output = data[..., indices]

    # ---------------------------------------------------------
    # Save if requested
    # ---------------------------------------------------------

    if output_file is not None:
        output_file = Path(output_file)

        if input_is_nifti:
            output_img = nib.Nifti1Image(
                output.astype(np.float32),
                affine=input_img.affine,
                header=input_img.header,
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

    return output, indices