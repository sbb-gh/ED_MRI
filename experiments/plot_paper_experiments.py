import numpy as np
from pathlib import Path
import models_simulations_fitting  # Assuming this module has your plotting functions

# Define paths and parameters
save_figs_dir = Path("/Users/paddyslator/python/ED/ED_MRI/examples/images_test")  # Replace with the actual directory path
experiment_names = ["NODDI_model","VERDICT_model","ADC_model","T1inv_model"]  # List the experiments as appropriate
SNR_all = [10, 20, 30, 40, 50]  # Replace with actual SNR levels


# Loop through each experiment and SNR to load data and plot
for experiment_name in experiment_names:
    results_plot = {}
    results_plot_transformed = {}

    # Load normalized data for bar plots - the largest SNR file contains all the SNRs
    results_plot_path = save_figs_dir / f"{experiment_name}_SNR{SNR_all[-1]}_predicted_vs_groundtruth_params_normalised.npy"
    results_plot = np.load(results_plot_path, allow_pickle=True).item()

    # Load untransformed data for predicted vs. ground truth plots
    results_plot_transformed_path = save_figs_dir / f"{experiment_name}_SNR{SNR_all[-1]}_predicted_vs_groundtruth_params.npy"
    results_plot_transformed = np.load(results_plot_transformed_path, allow_pickle=True).item()

    # Set experiment details for plotting, including save_figs_dir
    results_plot["experiment_name"] = experiment_name
    results_plot["SNR_all"] = SNR_all
    results_plot["save_figs_dir"] = save_figs_dir  # Ensure save_figs_dir is included here

    results_plot_transformed["experiment_name"] = experiment_name
    results_plot_transformed["SNR_all"] = SNR_all
    results_plot_transformed["save_figs_dir"] = save_figs_dir  # Ensure save_figs_dir is included here

    
    # Generate plots
    models_simulations_fitting.plot_barplots(results_plot)
    models_simulations_fitting.plot_predicted_vs_target_params(results_plot_transformed)
    models_simulations_fitting.plot_example_voxels(results_plot)

