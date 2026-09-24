""" (c) Stefano B. Blumberg and Paddy J. Slator, do not redistribute or modify"""
import logging
import timeit

from pathlib import Path

import sys

from model_fitting_network import ModelFittingTrainer
sys.path.append('/Users/paddyslator/python/ED/tadred')

import numpy as np
from tadred import tadred_main, utils

import os

import models_simulations_fitting
import models_simulations_plotting

log = logging.getLogger(__name__)
    

#save_figs_dir: str = '/home/blumberg/Bureau/z_Automated_Measurement/Output/journal_paper_tst/images' # None
#save_figs_dir: str = '/Users/paddyslator/python/ED/ED_MRI/examples/images_test' # None
save_dir: str = os.path.join(os.getcwd(), 'results', 'paper_experiments') # None

experiments = dict(
    #NODDI_model=models_simulations_fitting.NODDI,
    #VERDICT_model=models_simulations_fitting.VERDICT,
    SANDI_model=models_simulations_fitting.SANDI,
    #ADC_model=models_simulations_fitting.ADC,
    #T1inv_model=models_simulations_fitting.T1INV,
)

#hard code the model parameters and units for plot labels 
model_parameters = dict(
    #NODDI_model=('ODI','fstickinwatson', 'fiso', 'fwatson', 'n$_{x}$', 'n$_{y}$', 'n$_{z}$'), these are the pre-converted parameters
    #NODDI_model=('ODI','f$_{stick}$', 'f$_{ball}$', 'f$_{zeppelin}$', 'n$_{x}$', 'n$_{y}$', 'n$_{z}$'),
    #VERDICT_model=('R$_{sphere}$ ($\mu$m)', 'stick d$_{par}$ ($\mu$m s$^{-1}$)', 'f$_{sphere}$', 'f$_{ball}$','f$_{stick}$', 'n$_{x}$', 'n$_{y}$', 'n$_{z}$'),
    ADC_model=(r'ADC ($\mu$m$^2$ ms$^{-1}$)',),
    T1inv_model=('T1 (s)',),
    SANDI_model=('f$_{neurite}$','f$_{soma}$',r'D$_{neurite}$ ($\mu$m$^2$ ms$^{-1}$)',r'D$_{extra}$ ($\mu$m$^2$ ms$^{-1}$)',r'R$_{soma}$ ($\mu$m)',),
)

acquisition_param_name = dict(
    #NODDI_model='b-value (s $\mu$m$^{-2}$)',
    # VERDICT_model='b-value (s $\mu$m$^{-2}$)',
    ADC_model=r'b-value (s $\mu$m$^{-2}$)',
    T1inv_model='TI (s)',
    SANDI_model=r'b-value (s $\mu$m$^{-2}$)',
)

# num_samples: dict[str, int] = dict(
#     train=10**5,
#     val=10**4,
#     test=10**4,
# )
num_samples: dict[str, int] = dict(
    train=10**5,
    val=10**4,
    test=10**4,
)

#SNR_all: tuple[int,...] = (10, 20, 30, 40, 50)
SNR_all: tuple[int,...] = (10, 20)


# Neural network hyperparameters of the method TADRED
tadred_args = utils.load_base_args()
tadred_args.network.num_units_score: list[int] = [1000, 1000]
tadred_args.network.num_units_task: list[int] = [1000, 1000]
tadred_args.other_options.save_output = True
#tadred_args.tadred_train_eval.epochs = 50

#base filename for saving the trained model and results
tadred_args.output.out_base = save_dir
            

for experiment_name, experiment_cls in experiments.items():
    #save directory for the trained model and results    
    this_save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(this_save_dir, exist_ok=True)
    tadred_args.output.proj_name = experiment_name
    
    results_plot = dict(
        experiment_name=experiment_name, SNR_all=SNR_all, save_figs_dir=this_save_dir
    )
    
    results_plot_transformed = dict(
        experiment_name=experiment_name, SNR_all=SNR_all, save_figs_dir=this_save_dir
    )

    # Initialize a dictionary to store parameters for different splits
    fixed_params = {split: None for split in ("train", "val", "test")}

    for SNR in SNR_all:
        timer_SNR = timeit.default_timer()
        experiment = experiment_cls(SNR)
        # data: dict = dict()        
            
        tadred_args.output.run_name = experiment_name + "_SNR" + str(SNR) + "_n_train_vox_" + str(num_samples["train"]) 

        data = {}
        data_classical = {}
        
        for split in ("train", "val", "test"):

            if fixed_params[split] is None:

                experiment.create_params(
                    num_samples[split]
                )

                fixed_params[split] = (
                    experiment.params_for_model,
                    experiment.params_target,
                )

            (
                experiment.params_for_model,
                experiment.params_target,
            ) = fixed_params[split]

            # Dense / superdesign
            data[split] = (
                experiment.create_data_dense()
            )

            data[split + "_tar"] = (
                experiment.params_target
            )

            # CRLB / classical acquisition
            data_classical[split] = (
                experiment.create_data_classical()
            )

            data_classical[
                split + "_tar"
            ] = experiment.params_target


        # for split in ("train", "val", "test"):
        #     # Generate parameters only if not already generated - ensures that the same ground truth parameters are used for each SNR
        #     if fixed_params[split] is None:                            
        #         # Generate parameters and store both params_for_model and params_target
        #         experiment.create_params(num_samples[split])
        #         fixed_params[split] = (experiment.params_for_model, experiment.params_target)

        #     # Retrieve the stored parameters for the current split
        #     experiment.params_for_model, experiment.params_target = fixed_params[split]
                                    
        #     data_split = experiment.create_data_dense()
        #     data[split] = data_split
        #     data[split + "_tar"] = experiment.params_target
                                   

        #     if split == "test":
        #         data_classical_test = experiment.create_data_classical()

        feature_set_sizes_Ci = np.logspace(
            np.log(experiment.Cbar), np.log(experiment.Ceval), 5, base=np.exp(1), dtype=int
        )
        feature_set_sizes_Ci[0] = experiment.Cbar
        feature_set_sizes_Ci[-1] = experiment.Ceval
        tadred_args.tadred_train_eval.feature_set_sizes_Ci = [
            int(el) for el in feature_set_sizes_Ci
        ]
        tadred_args.tadred_train_eval.feature_set_sizes_evaluated = [int(experiment.Ceval)]
      
        tadred_result = tadred_main.run(tadred_args, data)

        #CRLB fitting and prediction using the classical acquisition scheme
        crlb_trainer = ModelFittingTrainer(
            hidden_units=list(
                tadred_args.network.num_units_task
            ),
            train_pytorch=tadred_args.train_pytorch,
            epochs=tadred_args.tadred_train_eval.epochs,
            no_gpu=tadred_args.other_options.no_gpu,
        )

        crlb_trainer.fit(
            train_x=data_classical["train"],
            train_y=data_classical["train_tar"],
            val_x=data_classical["val"],
            val_y=data_classical["val_tar"],
        )

        crlb_prediction = (
            crlb_trainer.predict(
                data_classical["test"]
            )
        )

        #DenseScheme fitting and prediction using the dense acquisition scheme
        dense_trainer = ModelFittingTrainer(
            hidden_units=list(
                tadred_args.network.num_units_task
            ),
            train_pytorch=tadred_args.train_pytorch,
            epochs=tadred_args.tadred_train_eval.epochs,
            no_gpu=tadred_args.other_options.no_gpu,
        )

        dense_trainer.fit(
            train_x=data["train"],
            train_y=data["train_tar"],
            val_x=data["val"],
            val_y=data["val_tar"],
        )

        dense_prediction = (
            dense_trainer.predict(
                data["test"]
            )
        )

        predictions = dict(
            CRLB=crlb_prediction,
            DenseScheme=dense_prediction,
            TADRED=tadred_result[
                experiment.Ceval
            ]["test_output"],
        )
        
        
        # predictions = dict(
        #     CRLB=experiment.fit_and_prediction(data_classical_test, "classical"),
        #     DenseScheme=experiment.fit_and_prediction(data["test"], "dense"),
        #     TADRED=tadred_result[experiment.Ceval]["test_output"],
        # )
              
        #example voxel for plotting
        example_voxel = dict(
            DenseScheme=data["test"][0,:],
            CRLB=data_classical["test"][0,:],            
            TADRED=data["test"][0,models_simulations_fitting.extract_tadred_index(tadred_result)],
        ) 
        #example part of the acquisition scheme for plotting, e.g. b-value, TI
        example_acquisition_param = dict(
            DenseScheme=experiment.extract_example_acquisition_param("dense"),
            CRLB=experiment.extract_example_acquisition_param("classical"),
            TADRED=experiment.extract_example_acquisition_param("tadred",tadred_result),
        )                              
        
                        
        results_plot[SNR] = dict(target=data["test_tar"], 
                                 predictions=predictions,
                                 example_acquisition_param=example_acquisition_param,
                                 example_voxel=example_voxel,
        )                    
              
        results_plot["parameter_labels"] = model_parameters[experiment_name]
        results_plot["acquisition_param_name"] = acquisition_param_name[experiment_name]
                
        print(f"Time for {SNR} is {timeit.default_timer() - timer_SNR} sec")


        #create a directory for the figure data
        figure_data_dir = Path(results_plot["save_figs_dir"], "figure_data")
        figure_data_dir.mkdir(parents=True, exist_ok=True)
        log.info("Figure data directory:", figure_data_dir)

        np.save(
            Path(
                figure_data_dir,
                f'{results_plot["experiment_name"]}_SNR{SNR}_predicted_vs_groundtruth_params_normalised.npy'  # Include the .npy extension
            ),
            results_plot  # This is the object to be saved
        )
        
            
        #for the predicted vs. ground truth plots, need to store the actual parameter values, not the normalised ones
        #undo the transformations to get the actual predicted parameter values
        predictions_transformed = dict(
            CRLB=experiment.params_target_to_model_input_params(predictions['CRLB']),
            DenseScheme=experiment.params_target_to_model_input_params(predictions['DenseScheme']),
            TADRED=experiment.params_target_to_model_input_params(predictions['TADRED']),
        )
        
        results_plot_transformed[SNR] = dict(
            experiment_name=experiment_name, SNR_all=SNR_all, save_figs_dir=this_save_dir
        )
        
        results_plot_transformed[SNR] = dict(target=experiment.params_target_to_model_input_params(data["test_tar"]), 
                                    predictions=predictions_transformed,)
            
        results_plot_transformed["parameter_labels"] =  model_parameters[experiment_name]

        #save the transformed results for the predicted vs. ground truth plots
        np.save(
            Path(
                figure_data_dir,
                f'{results_plot_transformed["experiment_name"]}_SNR{SNR}_predicted_vs_groundtruth_params.npy'  # Include the .npy extension
            ),
            results_plot_transformed  # This is the object to be saved
        )

        #create a directory for the simulation data
        sim_data_dir = Path(results_plot["save_figs_dir"], "data")
        sim_data_dir.mkdir(parents=True, exist_ok=True)
        log.info("Simulation data directory:", sim_data_dir)
        #save the whole simulation data in a big dictionary
        np.save(
            Path(
                sim_data_dir,
                f'{results_plot_transformed["experiment_name"]}_SNR{SNR}_all_simulated_data.npy'  # Include the .npy extension
            ),
            data  # This is the object to be saved
        )
        #also save the individual parts of the simulated data in separate files for easier access
        for split in ("train", "val", "test"):
            np.save(
                Path(
                    sim_data_dir,
                    f'{results_plot_transformed["experiment_name"]}_SNR{SNR}_{split}_simulated_signals.npy'  # Include the .npy extension
                ),
                data[split]  # This is the object to be saved
            )
        #and save the individual parts of the simulated target parameters in separate files for easier access
        for split in ("train", "val", "test"):
            np.save(
                Path(
                    sim_data_dir,
                    f'{results_plot_transformed["experiment_name"]}_SNR{SNR}_{split}_simulated_gt_params.npy'  # Include the .npy extension
                ),
                data[split + "_tar"]  # This is the object to be saved
            )

    #create a directory for the figures                
    figure_dir = Path(results_plot["save_figs_dir"], "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output figures directory:", figure_dir)
    
    #plot the barplots using the normalised data            
    models_simulations_plotting.plot_barplots(results_plot)
    
    #plot the predicted vs. ground truth plots using the untransformed data            
    models_simulations_plotting.plot_predicted_vs_target_params(results_plot_transformed)
        
    #plot the signal from one voxel for each acquistion scheme
    models_simulations_plotting.plot_example_voxels(results_plot)

        
    

    print("EOF")
