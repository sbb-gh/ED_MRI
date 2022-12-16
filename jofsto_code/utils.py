# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import numpy as np
import random, pickle
import os


def data_dict_to_array(data_dict, names, data_voxels=None, concat=True):
    """
    data_dict: dict each entry is a subject/split
    names: list of strings

    return numpy tuple array
    """

    names_inp = names.copy()
    data_out_inp = tuple([data_dict[name] for name in names_inp])
    data_out_inp = np.concatenate(data_out_inp, axis=0)
    data_out_inp = data_out_inp.astype(np.float32)
    print(
        "concatenated", names_inp, "for voxels", data_voxels, "on voxel dim", flush=True
    )

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


def load_data(data_fil):
    if data_fil.split(".")[-1] == "pkl":
        with open(data_fil, "rb") as f:
            data_dict = pickle.load(f)
    elif data_fil.split(".")[-1] == "npy":
        data_dict = np.load(data_fil, allow_pickle=True).item()
    else:
        assert False, "Data file either .pkl or .npy"
    return data_dict


def create_out_dirs(
    out_base,
    proj_base_name,
    proj_name,
):
    # if out_base is not None and len(proj_base_name) is not None:
    if len(out_base) > 0 and len(proj_base_name) > 0:
        out_base_dir = os.path.join(out_base, proj_base_name)
        os.makedirs(out_base_dir, exist_ok=True)
        print("Output base directory:", out_base_dir)
    else:
        out_base_dir = None
        print("Did not create output base directory")

    if len(out_base) > 0 and len(proj_base_name) > 0:
        # if out_base is not None and proj_base_name is not None:
        save_model_path = os.path.join(out_base_dir, proj_name)
        # os.makedirs(save_model_path,exist_ok=True)
        print("Model saved", save_model_path, flush=True)
    else:
        print("Did not create model saved dir", flush=True)
        save_model_path = None

    return out_base_dir, save_model_path


def create_train_val_test(
    data_fil,
    data_train_subjs,
    data_val_subjs,
    data_test_subjs,
    pass_data=None,
):
    if pass_data is None:
        data_dict = load_data(data_fil)
    else:
        data_dict = pass_data
    datatrain = data_dict_to_array(data_dict, data_train_subjs)
    dataval = data_dict_to_array(data_dict, data_val_subjs)
    datatest = data_dict_to_array(data_dict, data_test_subjs)

    return datatrain, dataval, datatest


def calc_affine_norm(
    datatrain,
    data_normalization,
    prctsig,  # Percentile for data normalization
    smallsig,  # Clip all values strictly less than
):

    if data_normalization == "original":
        max_val = np.float32(np.percentile(datatrain, prctsig))
    elif data_normalization == "original-measurement":
        max_val = np.float32(np.percentile(np.abs(datatrain), prctsig, axis=0))
    else:
        assert False

    assert (
        smallsig == 0
    ), "Rewrite for smallsig > 0 datatrain[datatrain<smallsig] = smallsig"
    min_val = smallsig
    loss_affine = (max_val - min_val, min_val)

    return loss_affine


def create_data_norm(
    data_fil,
    data_train_subjs,
    data_val_subjs,
    data_test_subjs,
    data_normalization="original-measurement",
    prctsig=99.0,
    smallsig=0.0,
    pass_data=None,
    **kwargs,
):

    datatrain, dataval, datatest = create_train_val_test(
        data_fil,
        data_train_subjs,
        data_val_subjs,
        data_test_subjs,
        pass_data=pass_data,
    )

    data = dict(
        train_x=datatrain[0],
        train_y=datatrain[1],
        val_x=dataval[0],
        val_y=dataval[1],
        test_x=datatest[0],
        test_y=datatest[1],
    )

    # assume data prepared correctly
    data["n_features"] = datatrain[0].shape[1]
    data["out_units"] = datatrain[1].shape[1]

    loss_affine_x = calc_affine_norm(
        datatrain[0], data_normalization, prctsig, smallsig
    )
    loss_affine_y = calc_affine_norm(
        datatrain[1], data_normalization, prctsig, smallsig
    )

    assert loss_affine_y[1] in (0, None)

    data["loss_affine_x"] = loss_affine_x
    data["loss_affine_y"] = loss_affine_y

    data["train_x_median"] = np.median(data["train_x"], axis=0)

    return data


def print_dict(dictionary):
    try:
        for key, val in dictionary.items():
            print(key + ":", val)
    except:
        for key, val in dictionary.__dict__.items():
            print(key + ":", val)


def save_results_dir(
    out_base_dir,
    results,
    run_name,
):
    if out_base_dir is not [None,""]:
        results_dir = os.path.join(out_base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        all_results_fil = os.path.join(results_dir, run_name + "_all.npy")

        print("Saving final results in", all_results_fil, flush=True)
        np.save(all_results_fil, results)
    else:
        print("Do not save final results")


def set_random_seed_tf(seed):
    """Set random seed for tensorflow, seed is int"""
    import tensorflow as tf

    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except:
        tf.compat.v1.set_random_seed(seed)  # SEFS
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
    framework="tf",  # {"tf","pt"}
):
    if framework == "tf":
        set_random_seed_tf(seed)
    elif framework == "pt":
        set_random_seed_pt(seed)
    print("Random seed is", seed)
