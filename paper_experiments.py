""" (c) Stefano B. Blumberg and Paddy J. Slator, do not redistribute or modify"""
import timeit

import numpy as np
from tadred import tadred_main, utils

import models_simulations_fitting

save_figs_dir: str = None

experiments = dict(
    NODDI_model=models_simulations_fitting.NODDI,
    VERDICT_model=models_simulations_fitting.VERDICT,
    # ADC_model=models_simulations_fitting.ADC,
    # T1inv_model=models_simulations_fitting.T1INV,
)

num_samples: dict[str, int] = dict(
    train=10**4,
    val=10**3,
    test=10**3,
)
SNR_all = (10, 20, 30, 40, 50)

# Neural network hyperparameters of the method TADRED
tadred_args = utils.load_base_args()
tadred_args.network.num_units_score = [1000, 1000]
tadred_args.network.num_units_task = [1000, 1000]
tadred_args.other_options.save_output = True


for experiment_name, experiment_cls in experiments.items():
    results_plot = dict(
        experiment_name=experiment_name, SNR_all=SNR_all, save_figs_dir=save_figs_dir
    )

    for SNR in SNR_all:
        timer_SNR = timeit.default_timer()
        experiment = experiment_cls(SNR)
        data: dict = dict()

        for split in ("train", "val", "test"):
            experiment.create_params(num_samples[split])
            data_split = experiment.create_data_dense()
            data[split] = data_split
            data[split + "_tar"] = experiment.params_target

            if split == "test":
                data_classical_test = experiment.create_data_classical()

        feature_set_sizes_Ci = np.logspace(
            np.log(experiment.Cbar), np.log(experiment.Ceval), 5, base=np.exp(1), dtype=int
        )
        feature_set_sizes_Ci[0] = experiment.Cbar
        feature_set_sizes_Ci[-1] = experiment.Ceval
        tadred_args.tadred_train_eval.feature_set_sizes_Ci = [
            int(el) for el in feature_set_sizes_Ci
        ]
        tadred_args.tadred_train_eval.feature_set_sizes_evaluated = [int(experiment.Ceval)]

        predictions = dict(
            CRLB=experiment.fit_and_prediction(data_classical_test, "classical"),
            DenseScheme=experiment.fit_and_prediction(data["test"], "dense"),
            TADRED=tadred_main.run(tadred_args, data)[experiment.Ceval]["test_output"],
        )

        results_plot[SNR] = dict(target=data["test_tar"], predictions=predictions)
        print(f"Time for {SNR} is {timeit.default_timer() - timer_SNR} sec")

    models_simulations_fitting.plot_predicted_vs_target_params(results_plot)
    models_simulations_fitting.plot_barplots(results_plot)

    print("EOF")
