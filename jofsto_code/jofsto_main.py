# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files

import argparse
import timeit

from .trainer_pt import Trainer
from .utils import create_data_norm, create_out_dirs, print_dict, save_results_dir, set_random_seed


def return_argparser():
    parser = argparse.ArgumentParser(description="JOFSTO")
    paradd = parser.add_argument
    paradd("--cfg", type=str, default="", help="Path to YAML config file", required=True)
    return parser


def run(args, pass_data=None):
    """Run JOFSTO with dict args and option to pass_data directly"""

    data, data_features_norm = create_data_norm(**args["data_norm"], pass_data=pass_data)
    assert data_features_norm["n_features"] == args["jofsto_train_eval"]["C_i_values"][0]
    out_dirs = create_out_dirs(**args["output"])
    print_dict(args)
    set_random_seed(seed=args["other_options"]["random_seed_value"], framework="pt")

    # Network structure hyperparameters and normalization args
    jofsto_network = dict(**args["network"], **data_features_norm)

    nnet = Trainer(
        jofsto_train_eval=args["jofsto_train_eval"],
        jofsto_network=jofsto_network,
        train_pytorch=args["train_pytorch"],
        other_options=args["other_options"],
    )

    start_train_timer = timeit.default_timer()
    results = nnet.train(**data)
    time_s = timeit.default_timer() - start_train_timer
    print(f"Total training time (s): {time_s} (h): {time_s / 3600}")

    results["args"] = args
    save_results_dir(out_dirs["out_base_dir"], out_dirs["results_fn"], results=results)

    return results
