# (c) Stefano B. Blumberg, do not redistribute or modify this file and helper files

import argparse
import timeit
from .trainer_pt import Trainer
from .utils import (
    create_data_norm,
    create_out_dirs,
    print_dict,
    save_results_dir,
    set_random_seed,
)


def return_argparser():
    parser = argparse.ArgumentParser(description="JOFSTO")
    paradd = parser.add_argument
    paradd("--cfg", type=str, default="", help="Path to YAML config file",required=True)
    return parser


def run(args, pass_data=None):

    assert args["epochs_fix_sigma"] + args["epochs_decay_sigma"] + args["epochs_decay"] < args["total_epochs"]
    data = create_data_norm(**args["data_norm"], pass_data=pass_data)
    out_base_dir, save_model_path = create_out_dirs(**args["output"])
    print_dict(args)
    set_random_seed(seed=args["random_seed_value"], framework="pt")

    ## Hyperparameters
    options = dict(
        no_gpu=args["no_gpu"],
        save_output=args["save_output"],
    )

    ## NAS hyperparameters
    jofsto_network = dict(
        **args["network"],
        n_features=data["n_features"], out_units=data["out_units"],
        train_x_median=data["train_x_median"],
        loss_affine_x=data["loss_affine_x"], loss_affine_y=data["loss_affine_y"],
    )

    update_params = dict(
        epochs=args["total_epochs"],
        epochs_decay=args["epochs_decay"],
        C_i_values=args["C_i_values"],
        save_model_path=save_model_path,
        epochs_fix_sigma=args["epochs_fix_sigma"],
        epochs_decay_sigma=args["epochs_decay_sigma"],
        C_i_eval=args["C_i_eval"],
        n_features=data["n_features"],
    )

    optimizer_params = dict(lr=args["learning_rate"])

    dataloader_params = dict(
        batch_size=args["batch_size"],
        num_workers=args["workers"],
        shuffle=True,
    )

    nnet = Trainer(
        save_model_path,
        update_params=update_params,
        jofsto_network=jofsto_network,
        dataloader_params=dataloader_params,
        optimizer_params=optimizer_params,
        options=options,
    )

    start_train_timer = timeit.default_timer()
    results = nnet.train(**data)

    time_s = timeit.default_timer() - start_train_timer
    print("Total training time (s):", time_s, "(h):", time_s / 3600)

    results["args"] = args
    results["data_test_subjs"] = args["data_norm"]["data_test_subjs"]
    results["C_i_eval"] = args["C_i_eval"]
    results["proj_name"] = args["output"]["proj_name"]
    results["run_name"] = args["output"]["run_name"]
    save_results_dir(out_base_dir=out_base_dir, results=results, run_name=args["output"]["run_name"])

    return results
