# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files

import pickle

import numpy as np


def load_data(data_fil):
    """Loads data dict from .pkl or .npy file."""
    if data_fil.split(".")[-1] == "pkl":
        with open(data_fil, "rb") as f:
            data_dict = pickle.load(f)
    elif data_fil.split(".")[-1] == "npy":
        data_dict = np.load(data_fil, allow_pickle=True).item()
    else:
        assert False, "Data file either .pkl or .npy"
    return data_dict


def data_dict_to_array(data_dict, names, data_voxels=None):
    """Conncatenate and prepare data for each split.

    Args:
    data_dict (dict): Each key is a subject
    names (List[str])

    Return:
    Input and target data
    """
    names_inp = names.copy()
    data_out_inp = tuple([data_dict[name] for name in names_inp])
    data_out_inp = np.concatenate(data_out_inp, axis=0)
    data_out_inp = data_out_inp.astype(np.float32)
    print("concatenated", names_inp, "for voxels", data_voxels, "on voxel dim", flush=True)

    names_tar = [name + "_tar" for name in names]
    try:
        data_out_tar = tuple([data_dict[name] for name in names_tar])
        data_out_tar = np.concatenate(data_out_tar, axis=0)
        data_out_tar = data_out_tar.astype(np.float32)
        print("Target data found", flush=1)
    except:
        data_out_tar = np.copy(data_out_inp)
        print("Target data set to input data", flush=1)

    return data_out_inp, data_out_tar


def create_train_val_test(
    data_fil,
    data_train_subjs,
    data_val_subjs,
    data_test_subjs,
    pass_data=None,
):
    """Creates three splits from loading data, or from passing data."""
    if pass_data is None:
        data_dict = load_data(data_fil)
    else:
        data_dict = pass_data

    datatrain = data_dict_to_array(data_dict, data_train_subjs)
    dataval = data_dict_to_array(data_dict, data_val_subjs)
    datatest = data_dict_to_array(data_dict, data_test_subjs)
    return datatrain, dataval, datatest


def calc_affine_norm(
    data_np,
    data_normalization,
):
    """Calculates constants for affine transformation of data.

    Args:
        data_np (np.array n_samples x input/target features): Data
        data_normalization ({"original"}): Normalization from paper

    Return:
        loss_affine (np.array,np.array):
            Normalize data with (data - loss_affine[1])/loss_affine[0]
    """

    if data_normalization == "original-measurement":
        prctsig = 99  # Percentile for calculating normalization
        smallsig = 0  # Clamp values below this to zero
        max_val = np.float32(np.percentile(np.abs(data_np), prctsig, axis=0))
    else:
        assert False

    assert smallsig == 0, "Rewrite for smallsig > 0 data_np[data_np<smallsig] = smallsig"
    min_val = smallsig

    loss_affine = (max_val - min_val, min_val)
    return loss_affine


def create_data_norm(
    data_fil,
    data_train_subjs,
    data_val_subjs,
    data_test_subjs,
    data_normalization="original-measurement",
    pass_data=None,
):
    """Process data and create splits.

    Args:
    data_norm: See config file

    Return:
    data: Processed data, input-target of splits
    data_features_norm: Other information/features of the data
    """

    datatrain, dataval, datatest = create_train_val_test(
        data_fil,
        data_train_subjs,
        data_val_subjs,
        data_test_subjs,
        pass_data=pass_data,
    )

    # Other preprocessing here

    data = dict(
        train_x=datatrain[0],
        train_y=datatrain[1],
        val_x=dataval[0],
        val_y=dataval[1],
        test_x=datatest[0],
        test_y=datatest[1],
    )

    # Assume data prepared correctly
    loss_affine_x = calc_affine_norm(datatrain[0], data_normalization)
    loss_affine_y = calc_affine_norm(datatrain[1], data_normalization)
    train_x_median = np.median(data["train_x"], axis=0)

    assert loss_affine_y[1] in (0, None)

    data_features_norm = dict(
        n_features=datatrain[0].shape[1],
        out_units=datatrain[1].shape[1],
        loss_affine_x=loss_affine_x,
        loss_affine_y=loss_affine_y,
        train_x_median=train_x_median,
    )

    return data, data_features_norm


def jofsto_data_format(train, val, test):
    """Helper function, tuple-np.array or np.array for each split to JOFSTO format"""
    data_inp = dict(train=train, val=val, test=test)
    data = dict()
    for split in ("train", "val", "test"):
        data_split = data_inp[split]
        if isinstance(data_split, tuple) and len(data_split) == 2:
            data[split] = data_split[0]
            data[split + "_tar"] = data_split[1]
        else:
            data[split] = data_split

    for key, val in data.items():
        assert isinstance(val, np.ndarray)
    return data
