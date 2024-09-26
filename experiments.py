import copy
import logging

import numpy as np
from matplotlib import pyplot as plt

from dmipy.core import modeling_framework  # type: ignore
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues  # type: ignore
from dmipy.data import saved_acquisition_schemes  # type: ignore
from dmipy.distributions import distribute_models  # type: ignore
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models  # type: ignore
from dmipy.utils import utils  # type: ignore

log = logging.getLogger(__name__)


# Dummy example base class for simulation experiments
class ExperimentsBase:
    def __init__(self):
        self.set_acquisition_scheme_super()
        self.set_acquisition_scheme_classical()
        self.performance = dict()
        self.predictions_all = dict(baseline=dict(), tadred=dict(), target=dict())

    def create_parameters(self, num_samples: int):
        raise NotImplementedError

    def set_acquisition_scheme_super(self):
        raise NotImplementedError

    def set_acquisition_scheme_classical(self):
        raise NotImplementedError

    def create_data_super(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def create_data_classical(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        raise NotImplementedError

    def compute_performance_and_cache(
        self,
        metrics: tuple[str],
        SNR: float,
        prediction_tadred: np.ndarray,
        prediction_baseline: np.ndarray,
        target: np.ndarray,
    ):
        self.performance[SNR] = dict()
        for metric in metrics:
            if metric == "MSE":
                tadred_mse = np.mean((prediction_tadred - target) ** 2)
                baseline_mse = np.mean((prediction_baseline - target) ** 2)
                self.performance[SNR][metric] = dict(tadred=tadred_mse, baseline=baseline_mse)

    def cache_for_plot(
        self,
        SNR: float,
        prediction_tadred: np.ndarray,
        prediction_baseline: np.ndarray,
        target: np.ndarray,
    ):
        self.predictions_all["baseline"][SNR] = prediction_baseline
        self.predictions_all["tadred"][SNR] = prediction_tadred
        self.predictions_all["target"][SNR] = target

    def plots(self):
        plot_all(self.predictions_all)


def add_noise(data, noise_scale: float) -> np.ndarray:
    """Add Rician noise to data"""
    rng = np.random.default_rng()
    rng.standard_normal(10, dtype=np.float32)

    # TODO add np.float32
    data_real = data + np.random.normal(scale=noise_scale, size=np.shape(data))
    data_imag = np.random.normal(scale=noise_scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy


def plot_predicted_vs_target_parameters(
    experiment_name: str,
    plot_data: dict[str, dict[str, str | np.ndarray]],
):
    SNR_all = tuple(plot_data.keys())

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    fits = ("CRLB", "superdesign", "TADRED")

    # SNR_all = (SNR,)
    fig, ax = plt.subplots(len(fits), len(SNR_all), figsize=[6 * len(SNR_all), 5 * len(fits)])
    fig.suptitle(f"{experiment_name}", fontsize=16)
    # for SNR_i, SNR in enumerate(SNR_all):
    for SNR_i, SNR in enumerate(SNR_all):
        target = plot_data[SNR].pop("target")
        for fit_i, (fit_name, fit) in enumerate(plot_data[SNR]):
            ax[fit_i, SNR_i].plot(target, fit, ".", markersize=1, color=colors[fit_i])

            if fit_i == 0:
                ax[fit_i, SNR_i].set_title(f"SNR = {SNR}", fontsize=32, weight="bold")

            xlim = (0, 1)
            ylim = (0, 1)  # Different for each experiment TODO an attr of exp class

            ax[fit_i, SNR_i].plot(xlim, ylim, "k", markersize=5)
            if fit_i == len(fits) - 1:
                ax[fit_i, SNR_i].set_xlabel(f"GT", fontsize=32)
            if SNR_i == 0:
                ax[fit_i, SNR_i].set_ylabel(
                    f"{fits[fit_i]} \n PRED",
                    fontsize=32,
                    color=colors[fit_i],
                )

            ax[fit_i, SNR_i].set_ylim(ylim)
            ax[fit_i, SNR_i].set_xlim(xlim)
        plot_data[SNR] = target
    print("a")


"""
j = 0
SNR = 10  # SNR[0]
for k, fit in enumerate(predictions):
    ax[k, j].plot(target, fit, ".", markersize=1, color=colors[k])

    if k == 0:
        ax[k, j].set_title(f"SNR = {SNR}", fontsize=32, weight="bold")

    ax[k, j].plot((0, 3.5), (0, 3.5), "k", markersize=5)
    if k == len(prediction_names) - 1:
        ax[k, j].set_xlabel("Ground truth D\n($\mu$m$^2$s$^{-1}$)", fontsize=32)
    if j == 0:
        ax[k, j].set_ylabel(
            prediction_names[k] + "\n predicted D\n($\mu$m$^2$s$^{-1}$)",
            fontsize=32,
            color=colors[k],
        )

    ax[k, j].set_ylim([0, 3.5])
    ax[k, j].set_xlim([0, 3.5])
"""


# Helper function for NODDI and VERDICT simulations
def add_cartesian(model_dmipy, parameters_dmipy: np.ndarray) -> np.ndarray:
    """Create parameters including spherical and Cartesian, parameters replacing spherical with Cartesian"""

    # mu_{name} is the name of the spherical coordinates to convert
    # Find mu parameter and therefore the index of theta and phi
    # TODO refactor below w: mu_index = next(i for i, s in enumerate(model.parameter_names) if "_mu" in s)
    mu_index = [i for i, s in enumerate(model_dmipy.parameter_names) if "_mu" in s][0]
    theta_phi_index = mu_index, mu_index + 1

    # Convert to cartesian coordinates
    mu_cartesian = utils.unitsphere2cart_Nd(parameters_dmipy[:, theta_phi_index])

    # Flip the direction of any cartesian points in the lower half of the sphere
    lower_index = mu_cartesian[:, 2] < 0
    mu_cartesian[lower_index, :] = -mu_cartesian[
        lower_index, :
    ]  # TODO check if can replace w mu_cartesian[lower_index, :] *= -1

    # Add cartesian coordinates to the parameter array
    parameters_spherical_and_cartesian = np.append(parameters_dmipy, mu_cartesian, axis=1)

    # Remove spherical coordinates ("mu") from the parameter
    parameters_cartesian_only = np.delete(
        parameters_spherical_and_cartesian, theta_phi_index, axis=1
    )

    return parameters_cartesian_only


class NODDI:
    def __init__(self):
        self.set_acquisition_scheme_super()
        self.set_acquisition_scheme_classical()
        self.create_model()

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

    def create_parameters(self, num_samples: int):
        self.num_samples = num_samples
        parameters_dict = dict(
            SD1WatsonDistributed_1_SD1Watson_1_mu=np.random.uniform(
                low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2)
            ),
            SD1WatsonDistributed_1_SD1Watson_1_odi=np.random.uniform(
                low=0.01, high=0.99, size=num_samples
            ),
            SD1WatsonDistributed_1_partial_volume_0=np.random.uniform(
                low=0.01, high=0.99, size=num_samples
            ),
            partial_volume_0=np.random.uniform(low=0.01, high=0.99, size=num_samples),
            partial_volume_1=1 - np.random.uniform(low=0.01, high=0.99, size=num_samples),
        )
        self.parameters = self.model.parameters_to_parameter_vector(**parameters_dict)

    def set_acquisition_scheme_super(self):
        self.acquisition_scheme_super = (
            saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
        )
        self.Cbar = self.acquisition_scheme_super.number_of_measurements

    def set_acquisition_scheme_classical(self):
        scheme_dict = dict(
            bvalues=np.squeeze(
                np.concatenate(
                    (np.tile(0, (9, 1)), np.tile(711, (30, 1)), np.tile(2855, (60, 1))), axis=0
                )
            ),
            delta=17.5,
            Delta=37.8,
            TE=78,
            gradient_directions=np.zeros(shape=(99, 3)),
        )

        # load the optimised gradient directions - have to cite 10.1002/mrm.24736!
        optimised_gradient_directions = np.loadtxt("noddi-optimal-direction-samples.txt")
        # rearrange into the two shells
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

    def process_data(self, signals, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = add_noise(signals, noise_scale)
        parameters_cartesian = add_cartesian(self.model, self.parameters)
        signals = signals.astype(np.float32)
        parameters_cartesian = parameters_cartesian.astype(np.float32)

        return signals, parameters_cartesian

    def create_data_super(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model.simulate_signal(self.acquisition_scheme_super, self.parameters)
        return self.process_data(signals, noise_scale)

    def create_data_classical(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model.simulate_signal(self.acquisition_scheme_classical, self.parameters)
        return self.process_data(signals, noise_scale)

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        model = copy.deepcopy(self.model)
        if scheme_name == "classical":
            acquisition_scheme = self.acquisition_scheme_classical
        elif scheme_name == "super":
            acquisition_scheme = self.acquisition_scheme_super
        else:
            raise ValueError("Pick scheme_name to be classical | super")
        model_fit = model.fit(acquisition_scheme=acquisition_scheme, data=data_test)
        parameters_cartesian_pred = add_cartesian(self.model, model_fit.fitted_parameters_vector)
        return parameters_cartesian_pred


class VERDICT:
    def __init__(self):
        self.set_acquisition_scheme_super()
        self.set_acquisition_scheme_classical()
        self.create_model()

    def create_model(self):
        """Create VERDICT Model https://pubmed.ncbi.nlm.nih.gov/25426656"""
        # Fix parameters and set ranges
        sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=0.9e-9)
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        self.model = modeling_framework.MultiCompartmentModel(models=[sphere, ball, stick])
        self.model.set_fixed_parameter("G1Ball_1_lambda_iso", 0.9e-9)
        self.model.set_parameter_optimization_bounds("C1Stick_1_lambda_par", [3.05e-9, 10e-9])

    def create_parameters(self, num_samples: int):
        # Random parameters wisth sensible upper and lower bounds
        # TODO add reference to table
        f_0 = np.random.uniform(low=0.01, high=0.99, size=num_samples)
        f_1 = np.random.uniform(low=0.01, high=0.99 - f_0, size=num_samples)
        f_2 = 1 - f_0 - f_1

        # Big parameter vector to simulate_signal
        self.parameters = self.model.parameters_to_parameter_vector(
            C1Stick_1_mu=np.random.uniform(
                low=[0, -np.pi], high=[np.pi, np.pi], size=(num_samples, 2)
            ),
            C1Stick_1_lambda_par=np.random.uniform(
                low=1e-9, high=10e-9, size=num_samples
            ),  # in m^2/s,
            S4SphereGaussianPhaseApproximation_1_diameter=np.random.uniform(
                low=0.01e-06, high=20e-06, size=num_samples
            ),
            partial_volume_0=f_0,
            partial_volume_1=f_1,
            partial_volume_2=f_2,
        )

    def set_acquisition_scheme_classical(self):
        # From https://cds.ismrm.org/protected/15MProceedings/PDFfiles/2872.pdf Table 1
        scheme_dict = dict(
            bvalues=np.array([3000, 2000, 1500, 500, 90]),
            delta=np.array([24.7, 13.2, 24.7, 12.2, 12.2]),
            Delta=np.array([43.8, 32.3, 43.4, 31.3, 23.8]),
            TE=np.array([90, 67, 90, 65, 50.0]),
            # gradient_strengths=np.array([0.0439,0.0758,0.0311,0.0415,0.0506]),
            gradient_directions=np.zeros(shape=(5, 3)),
        )

        # repeat schemes account for b0 values
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

    def set_acquisition_scheme_super(self):
        self.acquisition_scheme_super = (
            saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme()
        )
        self.Cbar = self.acquisition_scheme_super.number_of_measurements

    def verdict_params_norm(Parameters_VERDICT: np.ndarray) -> np.ndarray:
        """Normalize parameters to be approximately equal so evaluation will penalize incorrect prediction roughly same"""

        log.info("Normalizing means of Parameters_VERDICT")
        Parameters_VERDICT[:, 0] = Parameters_VERDICT[:, 0] * (10**5) * 0.5
        Parameters_VERDICT[:, 1] = Parameters_VERDICT[:, 1] * (10**8)

        return Parameters_VERDICT

    def process_data(self, signals, noise_scale) -> tuple[np.ndarray, np.ndarray]:
        signals = add_noise(signals, noise_scale)
        parameters_cartesian = add_cartesian(self.model, self.parameters)
        signals = signals.astype(np.float32)
        parameters_cartesian = parameters_cartesian.astype(np.float32)
        return signals, parameters_cartesian

    def create_data_super(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model.simulate_signal(self.acquisition_scheme_super, self.parameters)
        return self.process_data(signals, noise_scale)

    def create_data_classical(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model.simulate_signal(self.acquisition_scheme_classical, self.parameters)
        return self.process_data(signals, noise_scale)

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        model = copy.deepcopy(self.model)
        if scheme_name == "classical":
            acquisition_scheme = self.acquisition_scheme_classical
        elif scheme_name == "super":
            acquisition_scheme = self.acquisition_scheme_super
        else:
            raise ValueError("Pick scheme_name to be classical | super")
        model_fit = model.fit(acquisition_scheme=acquisition_scheme, data=data_test)
        parameters_cartesian_pred = add_cartesian(self.model, model_fit.fitted_parameters_vector)
        return parameters_cartesian_pred


class ADC:
    def __init__(self):
        self.set_acquisition_scheme_super()
        self.set_acquisition_scheme_classical()
        self.create_model()

    def create_model(self):
        # TODO put in lambda function
        def adc_model(D, bval):
            return np.exp(-bval * D)

        self.model = adc_model

    def create_parameters(self, num_samples: int):
        minD = 0.1
        maxD = 3
        self.parameters = np.random.uniform(low=minD, high=maxD, size=(num_samples, 1))

    def set_acquisition_scheme_super(self):
        def f_crlb(b, params, sigma):
            # params[0] is S0
            # params[1] is ADC
            # params = np.zeros(2)
            # params[0] = 1
            # params[1] = 1
            # sigma = 0.05
            # need 2 b-values - so assume there is always a b=0 (CRLB with 2 b-values always chooses a b=0 anyway)
            b = np.insert(b, 0, 0)

            dy = np.zeros((len(b), 2))
            dy[:, 0] = np.exp(-b * params[1])
            dy[:, 1] = -b * params[0] * np.exp(-b * params[1])

            fisher = (np.matmul(dy.T, dy)) / sigma**2

            invfisher = np.linalg.inv(fisher)
            # second diagonal element is the lower bound on the variance of the ADC
            f = invfisher[1, 1]
            return f

        # TODO check
        minb, maxb = 0, 5
        nb = 192
        self.acquisition_scheme_super = np.linspace(minb, maxb, nb)
        self.Cbar = len(self.acquisition_scheme_super)

    def set_acquisition_scheme_classical(self):
        # TODO think this needs to be loaded?
        # np.loadtxt(os.path.join(basedir, "crlb_code/crlb_adc_optimised_protocol.txt"))
        minb, maxb = 0, 5
        nb = 5
        self.acquisition_scheme_classical = np.linspace(minb, maxb, nb)

    def create_data_super(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model(self.parameters, self.acquisition_scheme_super)
        signals = add_noise(signals, noise_scale)
        return signals, self.parameters

    def create_data_classical(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model(self.parameters, self.acquisition_scheme_classical)
        signals = add_noise(signals, noise_scale)
        return signals, self.parameters

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        from scipy.optimize import minimize

        if scheme_name == "classical":
            acquisition_scheme = self.acquisition_scheme_classical
        elif scheme_name == "super":
            acquisition_scheme = self.acquisition_scheme_super
        else:
            raise ValueError("Pick scheme_name to be classical | super")

        def objective_function(D, bvals, signals):
            return np.mean((signals - self.model(D, bvals)) ** 2)

        num_test, num_measurements = data_test.shape
        Dstart = 1
        # TODO check
        out_all = []
        for i in range(num_test):
            out = minimize(
                objective_function,
                Dstart,
                args=(acquisition_scheme, data_test[i, :]),
                method="Nelder-Mead",
            )
            out_all.append(out.x.item())  # assumes single point solution
        return np.array(out_all)


class T1INV:
    def __init__(self):
        self.set_acquisition_scheme_super()
        self.set_acquisition_scheme_classical()
        self.create_model()

    def create_model(self):
        # TODO put in lambda function
        def t1_model(T1, ti, tr=7):
            signals = abs(1 - (2 * np.exp(-ti / T1)) + np.exp(-tr / T1))
            return signals

        self.model = t1_model

    def create_parameters(self, num_samples: int):
        minT1, maxT1 = 0.1, 7
        self.parameters = np.random.uniform(low=minT1, high=maxT1, size=(num_samples, 1))

    def set_acquisition_scheme_super(self):
        # TODO check
        minTi, maxTi = 0.1, 7
        self.Cbar = 192
        self.acquisition_scheme_super = np.linspace(minTi, maxTi, self.Cbar)

    def set_acquisition_scheme_classical(self):
        def f_crlb(ti, params, tr, sigma):
            # params[0] is S0
            # params[1] is T1
            # convert to R1
            params[1] = 1 / params[1]
            # tr = 7
            # sigma = 1

            dy = np.zeros((len(ti), 2))
            dy[:, 0] = 1 - 2 * np.exp(-ti * params[1]) + np.exp(-tr * params[1])
            dy[:, 1] = params[0] * (
                2 * ti * np.exp(-ti * params[1]) - tr * np.exp(-tr * params[1])
            )

            fisher = (np.matmul(dy.T, dy)) / sigma**2

            invfisher = np.linalg.inv(fisher)
            # second diagonal element is the lower bound on the variance of R1
            f = invfisher[1, 1]

            return f

    def create_data_super(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model(self.parameters, self.acquisition_scheme_super)
        signals = add_noise(signals, noise_scale)
        return signals, self.parameters

    def create_data_classical(self, noise_scale: float) -> tuple[np.ndarray, np.ndarray]:
        signals = self.model(self.parameters, self.acquisition_scheme_classical)
        signals = add_noise(signals, noise_scale)
        return signals, self.parameters

    def fit_and_prediction(self, data_test: np.ndarray, scheme_name: str) -> np.ndarray:
        # TODO sortout
        from scipy.optimize import minimize

        def objective_function(D, bvals, signals):
            return np.mean((signals - self.model(D, bvals)) ** 2)

        num_test, num_measurements = data_test.shape
        Dstart = 1
        # TODO check
        out_all = []
        for i in range(num_test):
            out = minimize(
                objective_function,
                Dstart,
                args=(self.acquisition_scheme_classical, data_test[i, :]),
                method="Nelder-Mead",
            )
            out_all.append(out.x.item())  # assumes single point solution
        return np.array(out_all)

        fitted_parameters_crlb[i] = minimize(
            rician_objective_function,
            paramstart,
            args=(acq_params_crlb, tr, signals_crlb[i, :], sigma_test),
            method="Nelder-Mead",
        ).x
