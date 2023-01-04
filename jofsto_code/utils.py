# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import os
import pickle
import random

import numpy as np
import yaml


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


def create_out_dirs(
    out_base,
    proj_name,
    run_name,
):
    """Create directories to save output if len(out_base) > 0.

    Output saved in <out_base>/<proj_name>/<run_name>/
    Results saved in <out_base>/results/<run_name>_all.npy
    """
    # if out_base is not None and len(proj_name) is not None:
    if len(out_base) > 0 and len(proj_name) > 0:
        out_base_dir = os.path.join(out_base, proj_name)
        os.makedirs(out_base_dir, exist_ok=True)
        print("Output base directory:", out_base_dir)
        results_dir = os.path.join(out_base_dir, "results")
        os.makedirs(out_base_dir, exist_ok=True)
        print("Output results directory:", out_base_dir)
        results_fn = os.path.join(results_dir, run_name + "_all.npy")
    else:
        out_base_dir = None
        results_fn = None
        print("Did not create output base directory")

    if len(out_base) > 0 and len(proj_name) > 0:
        # if out_base is not None and proj_name is not None:
        save_model_path = os.path.join(out_base_dir, run_name)
        # os.makedirs(save_model_path,exist_ok=True)
        print("Model saved", save_model_path, flush=True)
    else:
        print("Did not create model saved dir", flush=True)
        save_model_path = None

    out_dirs = dict(
        out_base_dir=out_base_dir,
        results_fn=results_fn,
        save_model_path=save_model_path,
    )
    return out_dirs


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


def load_yaml(file_path):
    """Loads .yaml config file from file_path on disk"""
    with open(file_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def load_results(
    full_path=None,
    out_base_dir=None,
    run_name=None,
):
    """Load results file from JOFSTO save.

    Option (i) Pass full path link
    Option (ii) Pass out_base_dir and run_name
    """
    # TODO cleanup
    if full_path is not None:
        load_path = full_path
    else:
        load_path = os.path.join(os.path.join(out_base_dir, "results"), run_name + "_all.npy")
    results_load = np.load(load_path, allow_pickle=True).item()
    return results_load


def print_dict(dictionary):
    """Print all key,val pairs from dict"""
    try:
        for key, val in dictionary.items():
            print(key + ":", val)
    except:
        for key, val in dictionary.__dict__.items():
            print(key + ":", val)


def save_results_dir(
    out_base_dir,
    results_fn,
    results,
):
    """Save final results file if requested"""
    if not out_base_dir in [None, ""]:
        print("Saving final results in", results_fn, flush=True)
        np.save(results_fn, results)
    else:
        print("Do not save final results")


def set_random_seed_tf(seed):
    """Set random seed for tensorflow, seed is int"""
    import tensorflow as tf

    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except:
        tf.compat.v1.set_random_seed(seed)
    random.seed(seed)


def set_random_seed_pt(seed):
    """Set random seed for tensorflow, seed is int"""
    import torch

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # for cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)


def set_random_seed(
    seed,
    framework="pt",
):
    """Set random seed (int) for Pytorch (pt) or Tensorflow (tf)"""
    if framework == "tf":
        set_random_seed_tf(seed)
    elif framework == "pt":
        set_random_seed_pt(seed)
    print(f"Random seed is {seed}")


def set_numpy_seed(seed):
    np.random.seed(seed)


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
