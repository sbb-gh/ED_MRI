""" (c) Stefano B. Blumberg and Paddy J. Slator, do not redistribute or modify"""

import models_simulations_fitting

from tadred import tadred_main, utils


experiments = dict(
    # NODDI=models_simulations_fitting.NODDI(),
    # VERDICt=models_simulations_fitting.VERDICT(),
    ADC=models_simulations_fitting.ADC(),
    # t1inv=models_simulations_fitting.T1INV,
)

num_samples: dict[str, int] = dict(
    train=10**2,
    val=10**1,
    test=10**1,
)
SNR_all = (10, 30, 50)

# Neural network hyperparameters of the method TADRED
tadred_args = utils.load_base_args()
tadred_args.network.num_units_score = [1000, 1000]
tadred_args.network.num_units_task = [1000, 1000]
tadred_args.other_options.save_output = True


for experiment_name, experiment in experiments.items():
    plot_data = dict(
        experiment_name=experiment_name, SNR_all=SNR_all, plot_args=experiment.plot_args()
    )
    for SNR in SNR_all:
        data_TADRED: dict = dict()
        for split in ("train", "val", "test"):
            experiment.create_parameters(num_samples[split])
            data_TADRED_split, parameters_target = experiment.create_data_super(
                noise_scale=1 / SNR
            )
            data_TADRED[split] = data_TADRED_split
            data_TADRED[split + "_tar"] = parameters_target

            if split == "test":
                data_classical_test, _ = experiment.create_data_classical(noise_scale=1 / SNR)

        eval_C = [experiment.Cbar // 16]
        tadred_args.tadred_train_eval.feature_set_sizes_Ci = [experiment.Cbar] + eval_C
        tadred_args.tadred_train_eval.feature_set_sizes_evaluated = eval_C

        predictions = dict(
            CRLB=experiment.fit_and_prediction(data_classical_test, "classical"),
            superdesign=experiment.fit_and_prediction(data_TADRED["test"], "super"),
            TADRED=tadred_main.run(tadred_args, data_TADRED)[eval_C[-1]]["test_output"],
        )

        plot_data[SNR] = dict(target=parameters_target, predictions=predictions)

    models_simulations_fitting.plot_predicted_vs_target_parameters(plot_data)
    models_simulations_fitting.plot_barplots(plot_data)

    print("EOF")
