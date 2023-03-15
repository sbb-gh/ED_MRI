# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files


import os
import random

import numpy as np
import yaml


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
        os.makedirs(results_dir, exist_ok=True)
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


def set_numpy_seed(seed):
    np.random.seed(seed)


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
