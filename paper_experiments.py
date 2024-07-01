""" (c) Stefano B. Blumberg and Paddy J. Slator, do not redistribute or modify"""

import models_simulations_fitting

from tadred import tadred_main, utils


experiments = dict(
    # NODDI_model=models_simulations_fitting.NODDI,
    # VERDICT_model=models_simulations_fitting.VERDICT,
    ADC_model=models_simulations_fitting.ADC,
    # T1inv_model=models_simulations_fitting.T1INV,
)

num_samples: dict[str, int] = dict(
    train=10**5,
    val=10**4,
    test=10**4,
)
SNR_all = (10, 20, 30)

# Neural network hyperparameters of the method TADRED
tadred_args = utils.load_base_args()
tadred_args.network.num_units_score = [1000, 1000]
tadred_args.network.num_units_task = [1000, 1000]
tadred_args.other_options.save_output = True


for experiment_name, experiment_cls in experiments.items():
    results_plot = dict(experiment_name=experiment_name, SNR_all=SNR_all)

    for SNR in SNR_all:
        experiment = experiment_cls(SNR)
        data: dict = dict()

        for split in ("train", "val", "test"):
            experiment.create_parameters(num_samples[split])
            data_split, parameters_target = experiment.create_data_super()
            data[split] = data_split
            data[split + "_tar"] = parameters_target

            if split == "test":
                data_classical_test, _ = experiment.create_data_classical()

        tadred_args.tadred_train_eval.feature_set_sizes_Ci = [experiment.Cbar, experiment.Ceval]
        tadred_args.tadred_train_eval.feature_set_sizes_evaluated = [experiment.Ceval]

        predictions = dict(
            CRLB=experiment.fit_and_prediction(data_classical_test, "classical"),
            superdesign=experiment.fit_and_prediction(data["test"], "super"),
            TADRED=tadred_main.run(tadred_args, data)[experiment.Ceval]["test_output"],
        )

        results_plot[SNR] = dict(target=parameters_target, predictions=predictions)

    results_plot["plot_args"] = experiment.plot_args()
    models_simulations_fitting.plot_predicted_vs_target_parameters(results_plot)
    models_simulations_fitting.plot_barplots(results_plot)

    print("EOF")
