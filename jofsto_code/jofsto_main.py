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
    paradd("--data_fil", type=str, default="", help="Path to data dictionary")
    paradd("--out_base", type=str, default="", help="Outputs saved directory")

    paradd("--data_train_subjs", type=str, nargs="*", default=["train"])
    paradd("--data_val_subjs", type=str, nargs="*", default=["val"])
    paradd("--data_test_subjs", type=str, nargs="*", default=["test"])
    paradd("--data_normalization", type=str, default="original-measurement")
    paradd(
        "--epochs_fix_sigma",
        type=int,
        default=25,
        help="Fix score after epoch, E_1 in paper",
    )
    paradd(
        "--epochs_decay_sigma",
        type=int,
        default=10,
        help="Progressively set score to be sample independent across number epochs, E_2 - E_1 in paper",
    )
    paradd(
        "--epochs_decay",
        "-e_d",
        type=int,
        default=10,
        help="Progressively modify mask across number epochs, E_3 - E_2 in paper",
    )
    paradd("--total_epochs", type=int, default=10000, help="E in paper")
    paradd("--learning_rate", type=float, default=0.0001)
    paradd("--batch_size", type=int, default=1500)
    paradd("--random_seed_value", type=int, default=0, help="Random seed value")
    paradd("--workers", type=int, default=0, help="Dataloader number of workers")
    paradd("--proj_name", type=str, default="tst", help="Output proj_name/run_name")
    paradd("--run_name", type=str, default="def", help="Output proj_name/run_name")
    paradd(
        "--C_i_values",
        nargs="*",
        type=int,
        help="Values of C_1, C_2,...",
    )
    paradd(
        "--C_i_eval",
        nargs="*",
        type=int,
        help="Evaluate at this C",
    )
    paradd(
        "--num_units_score",
        type=int,
        nargs="*",
        default=[1000],
        help="Intermediate units in Score Network S, [-1] to switch off",
    )
    paradd(
        "--num_units_task",
        type=int,
        nargs="*",
        default=[1000],
        help="Intermediate units in Task Network T, set to [-1] to switch off",
    )

    paradd(
        "--hcp_fit_parameters",
        action="store_true",
        help="Fit the model parameters on HCP data",
    )

    paradd(
        "--score_activation",
        type=str,
        default="doublesigmoid",
        help="Activation function for score \sigma in paper",
    )

    return parser


def run(args, pass_data=None):

    assert (
        args.epochs_fix_sigma + args.epochs_decay_sigma + args.epochs_decay
        < args.total_epochs
    )

    data = create_data_norm(**args.__dict__, pass_data=pass_data)
    out_base_dir, save_model_path = create_out_dirs(
        args.out_base, args.proj_name, args.run_name
    )
    print_dict(args)
    set_random_seed(seed=args.random_seed_value, framework="pt")

    ## Hyperparameters
    options = dict(
        out_base=args.out_base,
        proj_name=args.proj_name,
        run_name=args.run_name,
        hcp_fit_parameters=args.hcp_fit_parameters,
    )

    ## NAS hyperparameters
    network_params = dict(
        num_units_score=args.num_units_score,
        num_units_task=args.num_units_task,
        seed=args.random_seed_value,
        n_features=data["n_features"],
        out_units=data["out_units"],
        loss_affine_x=data["loss_affine_x"],
        loss_affine_y=data["loss_affine_y"],
        score_activation=args.score_activation,
        train_x_median=data["train_x_median"],
    )

    update_params = dict(
        epochs=args.total_epochs,
        epochs_decay=args.epochs_decay,
        n_features=data["n_features"],
        C_i_values=args.C_i_values,
        save_model_path=save_model_path,
        epochs_fix_sigma=args.epochs_fix_sigma,
        epochs_decay_sigma=args.epochs_decay_sigma,
        C_i_eval=args.C_i_eval,
    )

    optimizer_params = dict(lr=args.learning_rate)

    dataloader_params = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
    )

    nnet = Trainer(
        save_model_path,
        update_params=update_params,
        network_params=network_params,
        dataloader_params=dataloader_params,
        optimizer_params=optimizer_params,
        options=options,
    )

    start_train_timer = timeit.default_timer()
    results = nnet.train(**data)

    time_s = timeit.default_timer() - start_train_timer
    print("Total training time (s):", time_s, "(h):", time_s / 3600)

    results["args"] = args
    results["data_test_subjs"] = args.data_test_subjs
    results["C_i_eval"] = args.C_i_eval
    save_results_dir(out_base_dir=out_base_dir, results=results, run_name=args.run_name)

    print("EOF", __file__)


if __name__ == "__main__":
    parser = return_argparser()
    args = parser.parse_args()
    run(args)
