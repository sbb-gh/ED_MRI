import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def plot_predicted_vs_target_params(results_plot: dict[str, dict[str, str | np.ndarray]]):
    title_name = results_plot["experiment_name"].replace("_", " ").capitalize()
    SNR_all = results_plot["SNR_all"]
    # plot_lim = results_plot["plot_args"]["lim"]
    save_figs_dir = Path(results_plot["save_figs_dir"], "figures")
    save_figs_dir.mkdir(parents=True, exist_ok=True)

    for SNR_i, SNR in enumerate(SNR_all):
        target = results_plot[SNR]["target"]
        num_param = target.shape[1]
        num_pred = len(results_plot[SNR]["predictions"])
        fig, ax = plt.subplots(
            num_pred, num_param, figsize=[3 * num_param, 3 * num_pred], squeeze=False
        )
        fig.suptitle(
            f"{title_name} SNR {SNR}", fontsize=26
        )  # Predicted vs Ground Truth Parameters \n
        for param_i in range(num_param):
            #ax[0, param_i].set_title(f"Parameter {param_i}", fontsize=12)
            ax[0, param_i].set_title(results_plot["parameter_labels"][param_i], fontsize=12)
            for pred_i, (pred_name, pred_array) in enumerate(
                results_plot[SNR]["predictions"].items()
            ):                             
            
                if pred_array.ndim == 1:
                    pred_array_param = pred_array
                else:
                    pred_array_param = pred_array[:, param_i]
                                
                target_param = target[:, param_i]                                                 
                
                ax[pred_i, param_i].plot(
                    target_param, pred_array_param, ".", markersize=1, color=colors[pred_i]
                )

                plot_lim = (np.floor(min(target_param)), np.ceil(max(target_param)))
                ax[pred_i, param_i].plot(plot_lim, plot_lim, "k", markersize=5)
                ax[pred_i, param_i].set_ylim(plot_lim)
                ax[pred_i, param_i].set_xlim(plot_lim)

                if param_i == 0:
                    ax[pred_i, 0].set_ylabel(f"{pred_name}", fontsize=19, color=colors[pred_i])
            ax[num_pred - 1, param_i].set_xlabel(f"Ground Truth", fontsize=12)

        fig.savefig(
            Path(
                save_figs_dir,
                f'{results_plot["experiment_name"]}_SNR{SNR}_predicted_vs_groundtruth_params',
            ),
            bbox_inches="tight",
        )
        
        plt.close()



def plot_barplots(results_plot: dict[str, dict[str, str | np.ndarray]]):
    SNR_all = results_plot["SNR_all"]
    title_name = results_plot["experiment_name"].replace("_", " ").capitalize()
    save_figs_dir = Path(results_plot["save_figs_dir"], "figures")
    save_figs_dir.mkdir(parents=True, exist_ok=True)
    num_metrics = 2
    bar_width = 0.25
    fig, ax = plt.subplots(1, num_metrics, figsize=[4 * len(SNR_all), 6], squeeze=False)
    fig.suptitle(f"{title_name}", fontsize=32)

    for SNR_i, SNR in enumerate(SNR_all):
        target = results_plot[SNR]["target"]
        for pred_i, (pred_name, pred_array) in enumerate(results_plot[SNR]["predictions"].items()):
            MSE = np.mean((target - pred_array) ** 2)
            MAE = np.mean(np.abs(target - pred_array))
            
            bar_x_pos = pred_i * bar_width + SNR_i
            ax_args = dict(label=pred_name) if SNR_i == 0 else {}
            ax[0, 0].bar(bar_x_pos, MSE, width=bar_width, color=colors[pred_i], **ax_args)
            ax[0, 1].bar(bar_x_pos, MAE, width=bar_width, color=colors[pred_i], **ax_args)
            
    # Set font size for x-axis labels
    for subplot in [ax[0, 0], ax[0, 1]]:
        subplot.set_xticklabels(subplot.get_xticks(), fontsize=18)  # Adjust fontsize as needed

    ax[0, 0].set_ylabel("Mean Squared Error", fontsize=19)
    ax[0, 1].set_ylabel("Mean Absolute Error", fontsize=19)
    for metric_i in range(num_metrics):
        ax[0, metric_i].set_xticks([SNR_i + 1 * bar_width for SNR_i, SNR in enumerate(SNR_all)])
        ax[0, metric_i].set_xticklabels([f"SNR = {SNR}" for SNR in SNR_all])
        ax[0, metric_i].set_yscale("log")
        ax[0, metric_i].legend(fontsize=16)

    fig.savefig(Path(save_figs_dir, f'{results_plot["experiment_name"]}_barplot'))
    
    plt.close()

def plot_example_voxels(results_plot: dict[str, dict[str, str | np.ndarray]]):
    SNR_all = results_plot["SNR_all"]
    
    #hacky only plot some SNRs
    # SNR_all = [SNR_all[i] for i in [1, 3]]
    
    title_name = results_plot["experiment_name"].replace("_", " ").capitalize()
    save_figs_dir = Path(results_plot["save_figs_dir"], "figures")
    save_figs_dir.mkdir(parents=True, exist_ok=True)

    acquisition_param_name = results_plot["acquisition_param_name"] 
    
    fig, ax = plt.subplots(
            1, len(SNR_all), figsize=[6 * len(SNR_all), 6], squeeze=False
        )
    
    for SNR_i, SNR in enumerate(SNR_all):            
                                
        DenseScheme_acquisition_scheme = results_plot[SNR]["example_acquisition_param"]["DenseScheme"][:,0]
        CRLB_acquisition_scheme = results_plot[SNR]["example_acquisition_param"]["CRLB"][:,0]
        TADRED_acquisition_scheme = results_plot[SNR]["example_acquisition_param"]["TADRED"][:,0]                
            
        DenseScheme_example_voxel = results_plot[SNR]["example_voxel"]["DenseScheme"]
        CRLB_example_voxel = results_plot[SNR]["example_voxel"]["CRLB"]
        TADRED_example_voxel = results_plot[SNR]["example_voxel"]["TADRED"]
                        
        fig.suptitle(
            f"{title_name}", fontsize=26
        )  # Predicted vs Ground Truth Parameters \n
        
        #hardcoded fixes
        if results_plot["experiment_name"] == "VERDICT_model" or results_plot["experiment_name"] == "NODDI_model":
            DenseScheme_acquisition_scheme = DenseScheme_acquisition_scheme*1e-9
            CRLB_acquisition_scheme = CRLB_acquisition_scheme*1e-9
            TADRED_acquisition_scheme = TADRED_acquisition_scheme*1e-9
        
        ax[0, SNR_i].plot(DenseScheme_acquisition_scheme, DenseScheme_example_voxel, '.', color='grey', label='DenseScheme')
        ax[0, SNR_i].plot(TADRED_acquisition_scheme, TADRED_example_voxel, 'ro',markersize=7, label='TADRED')
        ax[0, SNR_i].plot(CRLB_acquisition_scheme, CRLB_example_voxel, 'bD',markersize=7,label='CRLB')

        
        ax[0, SNR_i].set_xlabel(f"{acquisition_param_name}", fontsize=19)
        ax[0, SNR_i].set_ylabel(f"Signal", fontsize=19)
        ax[0, SNR_i].set_title(f"SNR {SNR}", fontsize=19)
        
        if results_plot["experiment_name"] == "NODDI_model":
            ax[0, SNR_i].set_xlim([-0.2, 15])
        
        
        ax[0, SNR_i].legend(fontsize=16)
    
    
    fig.savefig(
        Path(
            save_figs_dir,
            f'{results_plot["experiment_name"]}_example_voxels',
        ),
        bbox_inches="tight",
    )
      
    plt.close()
    
    #option to plot with different shades and markers
    # from matplotlib.colors import LinearSegmentedColormap

    # # Define color maps for each acquisition type
    # blue_cmap = LinearSegmentedColormap.from_list("blues", ["lightblue", "blue"])
    # red_cmap = LinearSegmentedColormap.from_list("reds", ["lightcoral", "red"])
    # green_cmap = LinearSegmentedColormap.from_list("greens", ["lightgreen", "green"])

    # for SNR_i, SNR in enumerate(SNR_all):            
    #     DenseScheme_acquisition_scheme = results_plot[SNR]["example_acquisition_param"]["DenseScheme"][:, 0]
    #     CRLB_acquisition_scheme = results_plot[SNR]["example_acquisition_param"]["CRLB"][:, 0]
    #     TADRED_acquisition_scheme = results_plot[SNR]["example_acquisition_param"]["TADRED"][:, 0]                
        
    #     DenseScheme_example_voxel = results_plot[SNR]["example_voxel"]["DenseScheme"]
    #     CRLB_example_voxel = results_plot[SNR]["example_voxel"]["CRLB"]
    #     TADRED_example_voxel = results_plot[SNR]["example_voxel"]["TADRED"]

    #     # Find unique values in the second column for color mapping if present
    #     DenseScheme_vals = results_plot[SNR]["example_acquisition_param"]["DenseScheme"][:, 1] if results_plot[SNR]["example_acquisition_param"]["DenseScheme"].shape[1] > 1 else np.array([0])
    #     CRLB_vals = results_plot[SNR]["example_acquisition_param"]["CRLB"][:, 1] if results_plot[SNR]["example_acquisition_param"]["CRLB"].shape[1] > 1 else np.array([0])
    #     TADRED_vals = results_plot[SNR]["example_acquisition_param"]["TADRED"][:, 1] if results_plot[SNR]["example_acquisition_param"]["TADRED"].shape[1] > 1 else np.array([0])

    #     # Unique Delta values and corresponding color mapping
    #     unique_dense_vals = np.unique(DenseScheme_vals)
    #     unique_crlb_vals = np.unique(CRLB_vals)
    #     unique_tadred_vals = np.unique(TADRED_vals)

    #     dense_colors = [blue_cmap(i / (len(unique_dense_vals) - 1)) for i in range(len(unique_dense_vals))]
    #     crlb_colors = [red_cmap(i / (len(unique_crlb_vals) - 1)) for i in range(len(unique_crlb_vals))]
    #     tadred_colors = [green_cmap(i / (len(unique_tadred_vals) - 1)) for i in range(len(unique_tadred_vals))]

    #     fig.suptitle(f"{title_name}", fontsize=26)

    #     # Plot with specific shades and markers, and generate legend labels with Delta value
    #     for i, (scheme, voxel, val) in enumerate(zip(DenseScheme_acquisition_scheme, DenseScheme_example_voxel, DenseScheme_vals)):
    #         color = dense_colors[np.where(unique_dense_vals == val)[0][0]]
    #         label = f"DenseScheme, Δ = {val:.2g}" if i == 0 or DenseScheme_vals[i] != DenseScheme_vals[i - 1] else ""
    #         ax[0, SNR_i].scatter(scheme, voxel, color=color, marker='.', label=label)

    #     for i, (scheme, voxel, val) in enumerate(zip(CRLB_acquisition_scheme, CRLB_example_voxel, CRLB_vals)):
    #         color = crlb_colors[np.where(unique_crlb_vals == val)[0][0]]
    #         label = f"CRLB, Δ = {val:.2g}" if i == 0 or CRLB_vals[i] != CRLB_vals[i - 1] else ""
    #         ax[0, SNR_i].scatter(scheme, voxel, color=color, marker='x', label=label)
        
    #     for i, (scheme, voxel, val) in enumerate(zip(TADRED_acquisition_scheme, TADRED_example_voxel, TADRED_vals)):
    #         color = tadred_colors[np.where(unique_tadred_vals == val)[0][0]]
    #         label = f"TADRED, Δ = {val:.2g}" if i == 0 or TADRED_vals[i] != TADRED_vals[i - 1] else ""
    #         ax[0, SNR_i].scatter(scheme, voxel, color=color, marker='o', label=label)

    #     ax[0, SNR_i].set_xlabel(f"{acquisition_param_name}", fontsize=19)
    #     ax[0, SNR_i].set_ylabel("Signal", fontsize=19)
    #     ax[0, SNR_i].set_title(f"SNR {SNR}", fontsize=19)
        
        
    # # Create a legend in the first plot
    # ax[0, 0].legend(loc="upper right",fontsize=8)

    
        
    
        
