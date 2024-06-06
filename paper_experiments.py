""" (c) Stefano B. Blumberg and Paddy J. Slator, do not redistribute or modify"""

import simulations

from tadred import tadred_main, utils


experiments = dict(
    noddi=simulations.NODDI,
    # verdict=simulations.VERDICT,
    adc=simulations.ADC,
    # t1inv=simulations.T1INV,
)

num_samples: dict[str, int] = dict(
    train=10**2,
    val=10**1,
    test=10**1,
)

# Neural network hyperparameters of the method TADRED
tadred_args = utils.load_base_args()
tadred_args.network.num_units_score = [1000, 1000]
tadred_args.network.num_units_task = [1000, 1000]
tadred_args.other_options.save_output = True


for experiment_name in experiments:
    experiment = experiments[experiment_name]()

    for SNR in (10, 20, 30, 40, 50):
        data_TADRED: dict = dict()
        for split in ("train", "val", "test"):
            experiment.create_parameters(num_samples[split])
            data_TADRED_split, parameters_target = experiment.create_data_dense(noise_scale=1 / SNR)
            data_TADRED[split] = data_TADRED_split
            data_TADRED[split + "_tar"] = parameters_target

            if split == "test":
                data_baseline_test, _ = experiment.create_data_classical(noise_scale=1 / SNR)

        evaluation_C = [experiment.Cbar // 16]
        tadred_args.tadred_train_eval.feature_set_sizes_Ci = [experiment.Cbar] + evaluation_C
        tadred_args.tadred_train_eval.feature_set_sizes_evaluated = evaluation_C

        results_tadred = tadred_main.run(tadred_args, data_TADRED)
        prediction_baseline = experiment.classical_fitting_and_prediction(data_baseline_test)
        predictions_SNR = dict(
            prediction_baseline=prediction_baseline,
            prediction_tadred=results_tadred[evaluation_C[-1]]["test_output"],
            target=parameters_target,
            SNR=SNR,
        )

        # experiment.compute_performance_and_cache(metrics=('MSE',), **predictions_SNR)

        # TODO crap idea to cache all predictions across SNR...
        # experiment.cache_for_plot(**predictions_SNR)

    # experiment.plots()

    print("EOF")
