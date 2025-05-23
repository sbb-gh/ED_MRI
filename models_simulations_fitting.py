import copy
import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from dmipy.core import modeling_framework  # type: ignore
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues  # type: ignore
from dmipy.data import saved_acquisition_schemes  # type: ignore
from dmipy.distributions import distribute_models  # type: ignore
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models  # type: ignore
from dmipy.utils import utils  # type: ignore

colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def plot_predicted_vs_target_params(results_plot: dict[str, dict[str, str | np.ndarray]]):
    title_name = results_plot["experiment_name"].replace("_", " ").capitalize()
    SNR_all = results_plot["SNR_all"]
    # plot_lim = results_plot["plot_args"]["lim"]
    save_figs_dir = results_plot["save_figs_dir"]

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
    save_figs_dir = results_plot["save_figs_dir"]
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
    SNR_all = [SNR_all[i] for i in [1, 3]]
    
    title_name = results_plot["experiment_name"].replace("_", " ").capitalize()
    save_figs_dir = results_plot["save_figs_dir"]

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

    
        
    
        


       
        
  
    
def extract_tadred_index(tadred_result):
    #get the chosen acquisition scheme indices from the tadred results
    V_last = tadred_result["args"]["tadred_train_eval"]["feature_set_sizes_Ci"][-1]
    acq_params_tadred_index = tadred_result[V_last]["measurements"]

    return acq_params_tadred_index

def get_acquisition_scheme(experiment, scheme_name: str, tadred_result=None):  
    #get the acquisition scheme from the input string
    if scheme_name == "classical":
        acquisition_scheme = experiment.acquisition_scheme_classical
    elif scheme_name == "dense":
        acquisition_scheme = experiment.acquisition_scheme_dense  
    elif scheme_name == "tadred":
        acquisition_scheme = experiment.extract_tadred_optimized_scheme(tadred_result)
    else:
        raise ValueError("Pick scheme_name to be classical | dense | tadred")
    
    return acquisition_scheme

class SimulationsFitting:
    def __init__(self, SNR: float):
        self.SNR = SNR
        self.set_acquisition_scheme_dense()
        self.set_acquisition_scheme_classical()
        self.create_model()

    def create_model(self):
        # Define self.model, self.model_forward
        pass

    def create_params(self, num_samples: int) -> None:
        self.params_for_model: np.ndarray
        self.params_target: np.ndarray

    def set_acquisition_scheme_dense(self) -> None:
        self.acquisition_scheme_dense: any
        self.Cbar: int

    def set_acquisition_scheme_classical(self) -> None:
        self.acquisition_scheme_classical: any
        self.Ceval: int

    def create_data_dense(self) -> np.ndarray:
        signals = self.model_forward(self.acquisition_scheme_dense, self.params_for_model)
        signals = self.add_noise(signals, noise_scale=1 / self.SNR).astype(np.float32)
        return signals

    def create_data_classical(self) -> np.ndarray:
        signals = self.model_forward(self.acquisition_scheme_classical, self.params_for_model)
        signals = self.add_noise(signals, noise_scale=1 / self.SNR).astype(np.float32)
        return signals

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        pass

    def add_noise(self, data, noise_scale: float) -> np.ndarray:
        """Add Rician noise to data"""
        rng = np.random.default_rng()
        rng.standard_normal(10, dtype=np.float32)

        data_real = data + np.random.normal(scale=noise_scale, size=np.shape(data)).astype(
            np.float32
        )
        data_imag = np.random.normal(scale=noise_scale, size=np.shape(data)).astype(np.float32)
        data_noisy = np.sqrt(data_real**2 + data_imag**2)
        return data_noisy


class ADC(SimulationsFitting):
    def __init__(self, SNR: float):
        self.minb = 0
        self.maxb = 5
        self.minD = 0.1
        self.maxD = 3        
        super().__init__(SNR=SNR)

    def create_model(self):
        def adc_model(bval, D):
            return np.exp(-bval * D)

        self.model = adc_model
        self.model_forward = adc_model

    def create_params(self, num_samples: int):
        self.params_for_model = np.random.uniform(
            low=self.minD, high=self.maxD, size=(num_samples, 1)
        ).astype(np.float32)
        self.params_target = self.params_for_model

    def set_acquisition_scheme_dense(self) -> None:
        self.Cbar = 192
        self.acquisition_scheme_dense = np.linspace(self.minb, self.maxb, self.Cbar)

    def set_acquisition_scheme_classical(self) -> None:
        self.Ceval = self.Cbar // 16

        def f_crlb(b: np.ndarray, params: np.ndarray, sigma: float):
            # params[0] is S0
            # params[1] is ADC
            # params = np.zeros(2)
            # params[0] = 1
            # params[1] = 1
            # sigma = 0.05
            # need 2 b-values - so assume there is always a b=0 (CRLB with 2 b-values always chooses a b=0 anyway)
            b = np.insert(b, 0, 0)

            dy = np.zeros((len(b), 2), dtype=np.float32)
            dy[:, 0] = np.exp(-b * params[1])
            dy[:, 1] = -b * params[0] * np.exp(-b * params[1])

            fisher = (np.matmul(dy.T, dy)) / sigma**2

            invfisher = np.linalg.inv(fisher)
            # second diagonal element is the lower bound on the variance of the ADC
            f = invfisher[1, 1]
            return f

        # Calculate CRLB optimal acquisition parameter (e.g. b-value, TI) for a range of model parameters (e.g. ADC, T1)
        # Match number of model parameters in the range to the number of measurements in the final TADRED output

        # One less parameter for ADC as CRLB assumes a b=0
        params_init = np.linspace(0, self.maxD, self.Ceval)[1:]
        acq_params_crlb = []

        # Don't affect the optimisation so can be fixed
        S0 = 1
        sigma = 1 / self.SNR

        for i, param_init in enumerate(params_init):
            fixed_args = (np.array((S0, param_init)), sigma)  # (2,) float
            bnds = ((self.minb, self.maxb),)  # acq_params_crlb
            init = 1 / param_init  # int
            opt = minimize(f_crlb, init, args=fixed_args, method="Nelder-Mead", bounds=bnds).x
            acq_params_crlb.append(opt[0])
            
        acq_params_crlb.append(0) #for diffusion add a b=0 acquisition  
        self.acquisition_scheme_classical = np.array(acq_params_crlb)

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        if scheme_name == "classical":
            acquisition_scheme = self.acquisition_scheme_classical
        elif scheme_name == "dense":
            acquisition_scheme = self.acquisition_scheme_dense
        else:
            raise ValueError("Pick scheme_name to be classical | dense")

        def objective_function(D, bvals, signals):
            return np.mean((signals - self.model(bvals, D)) ** 2)

        def log_i0(x: np.ndarray):
            exact = x < 700
            approx = x >= 700

            lb0 = np.zeros(np.shape(x), dtype=np.float32)
            lb0[exact] = np.log(np.i0(x[exact]))
            # This is a more standard approximation.  For large x, I_0(x) -> exp(x)/sqrt(2 pi x).
            lb0[approx] = x[approx] - np.log(2 * np.pi * x[approx]) / 2

            return lb0

        def rician_log_likelihood(synth_signals: np.ndarray, signals: np.ndarray, sigma: float):
            sumsqsc = (signals**2 + synth_signals**2) / (2 * sigma**2)
            scp =  signals * synth_signals / sigma**2
            #    lb0 = np.log(np.i0(scp))
            lb0 = log_i0(scp)
            log_likelihoods = -2 * np.log(sigma) - sumsqsc + np.log(signals) + lb0
            return np.sum(log_likelihoods)

        def rician_objective_function(
            D: np.ndarray, bvals: np.ndarray, signals: np.ndarray, sigma: float
        ):
            return -rician_log_likelihood(self.model(bvals, D), signals, sigma)

        Dstart = 1
        # TODO check
        out_all = []
        for data_test_sample in data_test:
            out = minimize(
                rician_objective_function,
                Dstart,
                args=(acquisition_scheme, data_test_sample, 1 / self.SNR),
                method="Nelder-Mead",
            )
            out_all.append([out.x.item()])  # assumes single point solution
        return np.array(out_all)

    def params_target_to_model_input_params(self, params_target: np.ndarray) -> np.ndarray:                               
        return params_target
    
    def extract_tadred_optimized_scheme(self, tadred_result) -> np.ndarray:
        #get the chosen acquisition scheme indices from the tadred results
        acquisition_scheme_tadred_index = extract_tadred_index(tadred_result)
        
        acquisition_scheme_tadred = self.acquisition_scheme_dense[acquisition_scheme_tadred_index]
        
        return acquisition_scheme_tadred
    
    def extract_example_acquisition_param(self, scheme_name: str, tadred_result=None) -> np.ndarray:
        if scheme_name == "tadred" and tadred_result is None:
            raise ValueError("tadred_result must be provided when scheme_name is 'tadred'.")
        #extract the example acquisition parameters from the scheme - no further processing needed        
        example_acquisition_param = np.atleast_2d(get_acquisition_scheme(self, scheme_name, tadred_result)).reshape(-1, 1)
        
        return example_acquisition_param   
    
    def plot_args(self):
        return dict(lim=(0, 3.1))


class T1INV(SimulationsFitting):
    def __init__(self, SNR: float):
        self.minTi = 0.1
        self.maxTi = 7
        self.minT1 = 0.1
        self.maxT1 = 7
        super().__init__(SNR=SNR)




    def create_model(self):
        def t1_model(ti, T1, tr=7):
            return abs(1 - (2 * np.exp(-ti / T1)) + np.exp(-tr / T1))
            
        self.model = t1_model               
        self.model_forward = t1_model

    def create_params(self, num_samples: int):       
        self.params_for_model = np.random.uniform(
            low=self.minT1, high=self.maxT1, size=(num_samples, 1)
        ).astype(np.float32)
        self.params_target = self.params_for_model                        

    def set_acquisition_scheme_dense(self) -> None:    
        self.Cbar = 192
        self.acquisition_scheme_dense = np.linspace(self.minTi, self.maxTi, self.Cbar)

    def set_acquisition_scheme_classical(self) -> None:
        self.Ceval = self.Cbar // 16
        
        def f_crlb(ti: np.ndarray, params: np.ndarray, tr: float, sigma: float):
            # params[0] is S0, params[1] is T1
            # convert to R1
            params[1] = 1 / params[1]
            # tr = 7, sigma = 1

            dy = np.zeros((len(ti), 2), dtype=np.float32)
            dy[:, 0] = 1 - 2 * np.exp(-ti * params[1]) + np.exp(-tr * params[1])
            dy[:, 1] = params[0] * (
                2 * ti * np.exp(-ti * params[1]) - tr * np.exp(-tr * params[1])
            )

            fisher = (np.matmul(dy.T, dy)) / sigma**2
            
            invfisher = np.linalg.inv(fisher)
            # second diagonal element is the lower bound on the variance of R1
            f = invfisher[1, 1]

            return f
        
        # Calculate CRLB optimal acquisition parameter (i.e. TI) for a range of T1
        # Match number of model parameters in the range to the number of measurements in the final TADRED output
        params_init = np.linspace(0, self.maxT1, self.Ceval + 1)[1:]
        acq_params_crlb = []

        # Don't affect the optimisation so can be fixed
        S0 = 1
        sigma = 1 / self.SNR
        
        #define the tr
        tr = 7
        
        for i, param_init in enumerate(params_init):
            fixed_args = (np.array((S0, param_init)), tr, sigma)  # (2,) float
            bnds = ((self.minTi, self.maxTi),)  # acq_params_crlb                        
            init = param_init  # int
            opt = minimize(f_crlb, init, args=fixed_args, method="Nelder-Mead", bounds=bnds).x
            acq_params_crlb.append(opt[0])
            
        self.acquisition_scheme_classical = np.array(acq_params_crlb)
   
    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:             
        if scheme_name == "classical":
            acquisition_scheme = self.acquisition_scheme_classical
        elif scheme_name == "dense":
            acquisition_scheme = self.acquisition_scheme_dense
        else:
            raise ValueError("Pick scheme_name to be classical | dense")                
        
        def objective_function(T1, ti, tr, signals):
            return np.mean((signals - self.model(ti, T1, tr)) ** 2)

        def log_i0(x):
            exact = x < 700
            approx = x >= 700

            lb0 = np.zeros(np.shape(x))
            lb0[exact] = np.log(np.i0(x[exact]))
            # This is a more standard approximation.  For large x, I_0(x) -> exp(x)/sqrt(2 pi x).
            lb0[approx] = x[approx] - np.log(2 * np.pi * x[approx]) / 2

            return lb0
        
        def rician_log_likelihood(synth_signals: np.ndarray, signals: np.ndarray, sigma: float):
            sumsqsc = (signals**2 + synth_signals**2) / (2 * sigma**2)
            scp =  signals * synth_signals / sigma**2
            #    lb0 = np.log(np.i0(scp))
            lb0 = log_i0(scp)
            log_likelihoods = -2 * np.log(sigma) - sumsqsc + np.log(signals) + lb0
            return np.sum(log_likelihoods)
        
        def rician_objective_function(
            T1: np.ndarray, ti: np.ndarray, tr: np.ndarray, signals: np.ndarray, sigma: float
        ):
            return -rician_log_likelihood(self.model(ti, T1, tr), signals, sigma)

        T1start = 2
        # TODO check                
        out_all = []
        
        #define the tr
        tr = 7        
        
        for data_test_sample in data_test:
            out = minimize(
                rician_objective_function,
                T1start,
                args=(acquisition_scheme, tr, data_test_sample, 1 / self.SNR),
                method="Nelder-Mead",
            )     
            out_all.append([out.x.item()])  # assumes single point solution
        return np.array(out_all)
  
    def params_target_to_model_input_params(self, params_target: np.ndarray) -> np.ndarray:                               
        return params_target
    
    def extract_tadred_optimized_scheme(self, tadred_result) -> np.ndarray:
        #get the chosen acquisition scheme indices from the tadred results
        acquisition_scheme_tadred_index = extract_tadred_index(tadred_result)
        
        acquisition_scheme_tadred = self.acquisition_scheme_dense[acquisition_scheme_tadred_index]
        
        return acquisition_scheme_tadred
    
    def extract_example_acquisition_param(self, scheme_name: str, tadred_result=None) -> np.ndarray:
        if scheme_name == "tadred" and tadred_result is None:
            raise ValueError("tadred_result must be provided when scheme_name is 'tadred'.")
        #extract the example acquisition parameters from the scheme - no further processing needed
        example_acquisition_param = np.atleast_2d(get_acquisition_scheme(self, scheme_name, tadred_result)).reshape(-1, 1)
        
        return example_acquisition_param       
    
    def plot_args(self):
        return dict(lim=(0, 7.5))


class DMIPYModels(SimulationsFitting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_cartesian(self, model_dmipy, params_dmipy: np.ndarray) -> np.ndarray:
        """Create parameters including spherical and Cartesian, parameters replacing spherical with Cartesian."""

        # mu_{name} is the name of the spherical coordinates to convert
        # Find mu parameter and therefore the index of theta and phi
        # TODO refactor below w: mu_index = next(i for i, s in enumerate(model.parameter_names) if "_mu" in s)
        mu_index = [i for i, s in enumerate(model_dmipy.parameter_names) if "_mu" in s][0]
        theta_phi_index = mu_index, mu_index + 1

        #make sure the fibre directions all point in the same direction
        # params_dmipy[:, theta_phi_index] = self.flip_theta_phi(params_dmipy[:, theta_phi_index])

        # Convert to cartesian coordinates
        mu_cartesian = utils.unitsphere2cart_Nd(params_dmipy[:, theta_phi_index])

        # Flip the direction of any cartesian points in the lower half of the sphere
        lower_index = mu_cartesian[:, 2] < 0
        # # TODO check if can replace w mu_cartesian[lower_index, :] *= -1
        mu_cartesian[lower_index, :] = -mu_cartesian[lower_index, :]

        # Add cartesian coordinates to the parameter array
        params_spherical_and_cartesian = np.append(params_dmipy, mu_cartesian, axis=1)

        # Remove spherical coordinates ("mu") from the parameter
        params_cartesian_only = np.delete(params_spherical_and_cartesian, theta_phi_index, axis=1)

        return params_cartesian_only.astype(np.float32)
    
    # def flip_theta_phi(self, theta_phi: np.ndarray) -> np.ndarray:
    #     # If phi is negative, flip it to be positive and adjust theta accordingly
    #     flip_indices = theta_phi[:, 1] < 0
    #     theta_phi[flip_indices, 1] = -theta_phi[flip_indices, 1]
    #     theta_phi[flip_indices, 0] = (theta_phi[flip_indices, 0] + np.pi) % (2 * np.pi) # Flip the angle by adding pi and wrap it within [0, 2*pi)
    #     return theta_phi
    
    # def create_fibre_direction_same_direction_polar(self, num_samples: int) -> np.ndarray:
    #     # Generate random fibre directions always pointing upwards
    #     theta_phi = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2))
    #     # Flip the direction of any points in the lower half of the sphere         
    #     theta_phi = self.flip_theta_phi(theta_phi)                 
    #     return theta_phi


    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        model = copy.deepcopy(self.model)
        if scheme_name == "classical":
            acquisition_scheme = self.acquisition_scheme_classical
        elif scheme_name == "dense":
            acquisition_scheme = self.acquisition_scheme_dense
        else:
            raise ValueError("Pick scheme_name to be classical | dense")
        model_fit = model.fit(acquisition_scheme=acquisition_scheme, data=data_test, Ns=5, N_sphere_samples=30)        
                
        # params_target_prediction = self.add_cartesian(
        #     self.model, model_fit.fitted_parameters_vector
        # )

        # #make sure the fibre directions all point in the same direction
        # mu_index = [i for i, s in enumerate(model.parameter_names) if "_mu" in s][0]
        # theta_phi_index = mu_index, mu_index + 1
        # model_fit.fitted_parameters_vector[:, theta_phi_index] = self.flip_theta_phi(model_fit.fitted_parameters_vector[:, theta_phi_index])


        params_target_prediction = self.model_input_params_to_params_target(
            model_fit.fitted_parameters_vector
        )
        return params_target_prediction
    

    

    
    
    
        
        
        
        
        


class NODDI(DMIPYModels):
    def __init__(self, SNR: float):
        super().__init__(SNR=SNR)

    def create_model(self):
        # Create NODDI model from https://pubmed.ncbi.nlm.nih.gov/22484410
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()
        watson_dispersed_bundle = distribute_models.SD1WatsonDistributed(models=[stick, zeppelin])

        # Fix parameters, tortuosity constraints, isotropic diffusivity
        # TODO magic numbers like 1.7e-9 or 3e-9 should be variables instead with a meaningful name that you could use and change when needed.
        watson_dispersed_bundle.set_tortuous_parameter(
            "G2Zeppelin_1_lambda_perp", "C1Stick_1_lambda_par", "partial_volume_0"
        )
        watson_dispersed_bundle.set_equal_parameter(
            "G2Zeppelin_1_lambda_par", "C1Stick_1_lambda_par"
        )
        watson_dispersed_bundle.set_fixed_parameter("G2Zeppelin_1_lambda_par", 1.7e-9)
        self.model = modeling_framework.MultiCompartmentModel(
            models=[ball, watson_dispersed_bundle]
        )
        self.model.set_fixed_parameter("G1Ball_1_lambda_iso", 3e-9)
        self.model_forward = self.model.simulate_signal

    def model_input_params_to_params_target(self, params_for_model: np.ndarray) -> np.ndarray:
        params_target = self.add_cartesian(self.model, params_for_model)
        return params_target
    
    def params_target_to_model_input_params(self, params_target: np.ndarray) -> np.ndarray:
        params_for_model = params_target.copy()
                   
        #convert volume fractions into single stick and zeppelin volume fractions       
        f_stickinwatson = params_for_model[:, 1] #normalized volume fraction of the Stick within the WatsonBundle.
        f_watson = params_for_model[:, 3] #volume fraction of the Watson bundle.

        #get stick volume fraction - partial_volume_1 * SD1WatsonDistributed_1_partial_volume_0
        params_for_model[:, 1] = f_watson * f_stickinwatson 
        #get zeppelin volume fraction - partial_volume_1 * (1 - SD1WatsonDistributed_1_partial_volume_0)
        params_for_model[:, 3] = f_watson * (1 - f_stickinwatson)
        
                    
        
        return params_for_model

    def create_params(self, num_samples: int):
        self.num_samples = num_samples
        
        f_ball = np.random.uniform(low=0.01, high=0.99, size=num_samples)
        f_watson = 1 - f_ball
        f_stickinwatson = np.random.uniform(low=0.01, high=0.99, size=num_samples)            
        
        params_dict = dict(
            SD1WatsonDistributed_1_SD1Watson_1_mu=np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2)),
            SD1WatsonDistributed_1_SD1Watson_1_odi=np.random.uniform(
                low=0.01, high=0.99, size=num_samples
            ),
            SD1WatsonDistributed_1_partial_volume_0 = f_stickinwatson,
            partial_volume_0=f_ball,
            partial_volume_1=f_watson,
        )
        self.params_for_model = self.model.parameters_to_parameter_vector(**params_dict).astype(
            np.float32
        )

        self.params_target = self.model_input_params_to_params_target(self.params_for_model)

    def set_acquisition_scheme_dense(self) -> None:
        self.acquisition_scheme_dense = (
            saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
        )
        self.Cbar = self.acquisition_scheme_dense.number_of_measurements

    def set_acquisition_scheme_classical(self) -> None:
        scheme_dict = dict(
            bvalues=np.squeeze(
                np.concatenate(
                    (np.tile(0, (9, 1)), np.tile(711, (30, 1)), np.tile(2855, (60, 1))), axis=0
                )
            ),
            delta=17.5,
            Delta=37.8,
            TE=78,
            gradient_directions=np.zeros(shape=(99, 3), dtype=np.float32),
        )

        # Load the optimised gradient directions - have to cite 10.1002/mrm.24736!
        optimised_gradient_directions = np.loadtxt("noddi-optimal-direction-samples.txt")
        # Rearrange into the two shells
        shell_1_gradient_directions = optimised_gradient_directions[
            optimised_gradient_directions[:, 0] == 1, 1:4
        ]
        shell_2_gradient_directions = optimised_gradient_directions[
            optimised_gradient_directions[:, 0] == 2, 1:4
        ]

        scheme_dict["gradient_directions"][9:39, :] = shell_1_gradient_directions
        scheme_dict["gradient_directions"][39:100, :] = shell_2_gradient_directions

        scheme_dict.update(dict(min_b_shell_distance=50e6, b0_threshold=10e6))

        scheme_dict["delta"] /= 1000
        scheme_dict["Delta"] /= 1000
        scheme_dict["TE"] /= 1000
        scheme_dict["bvalues"] *= 10**6

        self.acquisition_scheme_classical = acquisition_scheme_from_bvalues(**scheme_dict)
        self.Ceval = self.acquisition_scheme_classical.number_of_measurements
        
        
    def extract_tadred_optimized_scheme(self, tadred_result) -> np.ndarray:
        #get the chosen acquisition scheme indices from the tadred results
        acquisition_scheme_tadred_index = extract_tadred_index(tadred_result)
        
        #make the b0_threshold 3e9 to avoid any errors if tadred doesn't choose a b=0
        acquisition_scheme_tadred = acquisition_scheme_from_bvalues(bvalues=self.acquisition_scheme_dense.bvalues[acquisition_scheme_tadred_index], 
                                                                    delta=self.acquisition_scheme_dense.delta[acquisition_scheme_tadred_index], 
                                                                    Delta=self.acquisition_scheme_dense.Delta[acquisition_scheme_tadred_index], 
                                                                    gradient_directions=self.acquisition_scheme_dense.gradient_directions[acquisition_scheme_tadred_index,:],
                                                                    b0_threshold=3e9)
                            
                            
        return acquisition_scheme_tadred
    
    def extract_example_acquisition_param(self, scheme_name: str, tadred_result=None) -> np.ndarray:
        if scheme_name == "tadred" and tadred_result is None:
            raise ValueError("tadred_result must be provided when scheme_name is 'tadred'.")
        #extract the example acquisition parameters from the scheme
        acquisition_scheme = get_acquisition_scheme(self, scheme_name, tadred_result)
        
        example_acquisition_param = np.atleast_2d(1e-9*acquisition_scheme.bvalues).reshape(-1, 1)
        
        return example_acquisition_param


class VERDICT(DMIPYModels):
    def __init__(self, SNR: float):
        super().__init__(SNR=SNR)

    def create_model(self):
        """Create VERDICT Model https://pubmed.ncbi.nlm.nih.gov/25426656"""
        # Fix parameters and set ranges
        sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=0.9e-9)
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        self.model = modeling_framework.MultiCompartmentModel(models=[sphere, ball, stick])
        self.model.set_fixed_parameter("G1Ball_1_lambda_iso", 0.9e-9)
        self.model.set_parameter_optimization_bounds("C1Stick_1_lambda_par", [3e-9, 10e-9])
        self.model.set_parameter_optimization_bounds("S4SphereGaussianPhaseApproximation_1_diameter", [0.01e-06, 30e-06])
        self.model_forward = self.model.simulate_signal

    def model_input_params_to_params_target(self, params_for_model: np.ndarray) -> np.ndarray:
        params_target = self.add_cartesian(self.model, params_for_model)
        # Normalize parameters to be approximately equal so evaluation will penalize incorrect prediction roughly same
        params_target[:, 0] = params_target[:, 0] * (10**5) * 0.5
        params_target[:, 1] = params_target[:, 1] * (10**8)
        return params_target    
    
    def params_target_to_model_input_params(self, params_target: np.ndarray) -> np.ndarray:        
        params_for_model = params_target.copy()
        # Invert the transformation applied to each parameter - leave cartesian parameters
        params_for_model[:, 0] = params_for_model[:, 0] / (10**5 * 0.5)
        params_for_model[:, 1] = params_for_model[:, 1] / (10**8)        
        
        #convert sphere diameter to radius
        params_for_model[:, 0] = params_for_model[:, 0] / 2
        
        #convert to nicer units 
        #sphere radius from m to um
        params_for_model[:, 0] = params_for_model[:, 0] * 10**6
        #diffusivity from m^2/s to um^2/ms
        params_for_model[:, 1] = params_for_model[:, 1] * 10**9
                
        return params_for_model

    def create_params(self, num_samples: int):
        # Random parameters with sensible upper and lower bounds
        # TODO add reference to table
        #mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2))
        mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2))
        lambda_par = np.random.uniform(low=3e-9, high=10e-9, size=num_samples)  # in m^2/s
        diameter = np.random.uniform(low=0.01e-06, high=30e-06, size=num_samples)
        f_0 = np.random.uniform(low=0.01, high=0.99, size=num_samples)
        f_1 = np.random.uniform(low=0.01, high=0.99 - f_0, size=num_samples)
        f_2 = np.maximum(0, 1 - f_0 - f_1)

        # Big parameter vector to simulate_signal
        self.params_for_model = self.model.parameters_to_parameter_vector(
            C1Stick_1_mu=mu,
            C1Stick_1_lambda_par=lambda_par,
            S4SphereGaussianPhaseApproximation_1_diameter=diameter,
            partial_volume_0=f_0,
            partial_volume_1=f_1,
            partial_volume_2=f_2,
        ).astype(np.float32)
        self.params_target = self.model_input_params_to_params_target(self.params_for_model)

    def set_acquisition_scheme_classical(self) -> None:
        # From https://cds.ismrm.org/protected/15MProceedings/PDFfiles/2872.pdf Table 1
        scheme_dict = dict(
            bvalues=np.array([3000, 2000, 1500, 500, 90]),
            delta=np.array([24.7, 13.2, 24.7, 12.2, 12.2]),
            Delta=np.array([43.8, 32.3, 43.4, 31.3, 23.8]),
            TE=np.array([90, 67, 90, 65, 50.0]),
            # gradient_strengths=np.array([0.0439,0.0758,0.0311,0.0415,0.0506]),
            gradient_directions=np.zeros(shape=(5, 3), dtype=np.float32),
        )

        # Repeat schemes account for b0 values
        for key, val in scheme_dict.items():
            scheme_dict[key] = np.repeat(val, 4, axis=0)

        scheme_dict["bvalues"][0::4,] = 0
        scheme_dict["gradient_directions"][1::4,] = np.array([1, 0, 0])
        scheme_dict["gradient_directions"][2::4,] = np.array([0, 1, 0])
        scheme_dict["gradient_directions"][3::4,] = np.array([0, 0, 1])

        scheme_dict.update(dict(min_b_shell_distance=50e6, b0_threshold=10e6))

        scheme_dict["delta"] /= 1000
        scheme_dict["Delta"] /= 1000
        scheme_dict["TE"] /= 1000
        scheme_dict["bvalues"] *= 10**6
        self.acquisition_scheme_classical = acquisition_scheme_from_bvalues(**scheme_dict)
        self.Ceval = self.acquisition_scheme_classical.number_of_measurements

    def set_acquisition_scheme_dense(self) -> None:
        self.acquisition_scheme_dense = (
            saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme()
        )
        self.Cbar = self.acquisition_scheme_dense.number_of_measurements    
    
          
    def extract_tadred_optimized_scheme(self, tadred_result) -> np.ndarray:
        #get the chosen acquisition scheme indices from the tadred results
        acquisition_scheme_tadred_index = extract_tadred_index(tadred_result)
        
        #make the b0_threshold 3e9 to avoid any errors if tadred doesn't choose a b=0
        acquisition_scheme_tadred = acquisition_scheme_from_bvalues(bvalues=self.acquisition_scheme_dense.bvalues[acquisition_scheme_tadred_index], 
                                                                    delta=self.acquisition_scheme_dense.delta[acquisition_scheme_tadred_index], 
                                                                    Delta=self.acquisition_scheme_dense.Delta[acquisition_scheme_tadred_index], 
                                                                    gradient_directions=self.acquisition_scheme_dense.gradient_directions[acquisition_scheme_tadred_index,:],
                                                                    b0_threshold=3e9)
                        
        return acquisition_scheme_tadred
    
    def extract_example_acquisition_param(self, scheme_name: str, tadred_result=None) -> np.ndarray:
        if scheme_name == "tadred" and tadred_result is None:
            raise ValueError("tadred_result must be provided when scheme_name is 'tadred'.")
        #extract the example acquisition parameters from the scheme
        acquisition_scheme = get_acquisition_scheme(self, scheme_name, tadred_result)
        
        example_acquisition_param = np.column_stack((1e-9*acquisition_scheme.bvalues, acquisition_scheme.Delta))
        
        return example_acquisition_param
    
    def plot_args(self):
        return dict(lim=(0, 1))
