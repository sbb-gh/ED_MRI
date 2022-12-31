import numpy as np
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.utils.utils import unitsphere2cart_Nd


def noddi_model():
    """Create NODDI model from https://pubmed.ncbi.nlm.nih.gov/22484410"""

    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

    # Fix parameters and tortuosity constraints
    watson_dispersed_bundle.set_tortuous_parameter(
        "G2Zeppelin_1_lambda_perp", "C1Stick_1_lambda_par", "partial_volume_0"
    )
    watson_dispersed_bundle.set_equal_parameter("G2Zeppelin_1_lambda_par", "C1Stick_1_lambda_par")
    watson_dispersed_bundle.set_fixed_parameter("G2Zeppelin_1_lambda_par", 1.7e-9)

    NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    # Fix isotropic diffusivity
    NODDI_mod.set_fixed_parameter("G1Ball_1_lambda_iso", 3e-9)

    return NODDI_mod


def noddi(nsamples, acq_scheme):
    """Create NODDI model and simulate data

    Args:
        nsamples (int): Number of samples (voxels) to simulate
        acq_scheme (dmipy.data.saved_acquisition_schemes)

    Returns:
        Signals_NODDI: Simulated MRI signal
        Parameters_NODDI: Parameters used to simulate signal
    """

    NODDI_mod = noddi_model()

    mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(nsamples, 2))
    odi = np.random.uniform(low=0.01, high=0.99, size=nsamples)
    f_0 = np.random.uniform(low=0.01, high=0.99, size=nsamples)
    f_1 = 1 - f_0
    f_bundle = np.random.uniform(low=0.01, high=0.99, size=nsamples)

    # Big parameter vector for simulate_signal
    Parameters_NODDI_dmipy = NODDI_mod.parameters_to_parameter_vector(
        SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
        SD1WatsonDistributed_1_SD1Watson_1_odi=odi,
        SD1WatsonDistributed_1_partial_volume_0=f_bundle,
        partial_volume_0=f_0,
        partial_volume_1=f_1,
    )

    print("Simulating NODDI data", flush=True)
    Signals_NODDI = NODDI_mod.simulate_signal(acq_scheme, Parameters_NODDI_dmipy)
    # NODDI_mod.parameter_names
    Signals_NODDI = add_noise(Signals_NODDI)

    # add cartesian parameters parameters_NODDI are the parameters to learn
    Parameters_NODDI_all, Parameters_NODDI = add_cartesian(NODDI_mod, Parameters_NODDI_dmipy)

    Signals_NODDI = Signals_NODDI.astype(np.float32)
    Parameters_NODDI = Parameters_NODDI.astype(np.float32)
    Parameters_NODDI_dmipy = Parameters_NODDI_dmipy.astype(np.float32)
    Parameters_NODDI_all = Parameters_NODDI_all.astype(np.float32)

    return Signals_NODDI, Parameters_NODDI


def verdict_model():
    """Create VERDICT Model https://pubmed.ncbi.nlm.nih.gov/25426656"""

    # Fixed to verdict value
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=0.9e-9)
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    VERDICT_mod = MultiCompartmentModel(models=[sphere, ball, stick])

    # VERDICT fix parameter and set ranges
    VERDICT_mod.set_fixed_parameter("G1Ball_1_lambda_iso", 0.9e-9)
    VERDICT_mod.set_parameter_optimization_bounds("C1Stick_1_lambda_par", [3.05e-9, 10e-9])

    return VERDICT_mod


def verdict_params_norm(Parameters_VERDICT):
    """Normalize VERDICT parameters so all on same scale"""

    print("Normalizing means of Parameters_VERDICT")
    Parameters_VERDICT[:, 0] = Parameters_VERDICT[:, 0] * (10**5) * 0.5
    Parameters_VERDICT[:, 1] = Parameters_VERDICT[:, 1] * (10**8)

    return Parameters_VERDICT


def verdict(nsamples, acq_scheme):
    """Create VERDICT model and simulate data

    Args:
        nsamples (int): Number of samples (voxels) to simulate
        acq_scheme (dmipy.data.saved_acquisition_schemes)

    Returns:
        Signals_VERDICT: Simulated MRI signal
        Parameters_VERDICT: Parameters used to simulate signal
    """

    VERDICT_mod = verdict_model()

    # Random parameters with sensible upper and lower bounds
    mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(nsamples, 2))
    lambda_par = np.random.uniform(low=1e-9, high=10e-9, size=nsamples)  # in m^2/s
    diameter = np.random.uniform(low=0.01e-06, high=20e-06, size=nsamples)
    f_0 = np.random.uniform(low=0.01, high=0.99, size=nsamples)
    f_1 = np.random.uniform(low=0.01, high=0.99 - f_0, size=nsamples)
    f_2 = 1 - f_0 - f_1

    # Big parameter vector to simulate_signal
    Parameters_VERDICT_dmipy = VERDICT_mod.parameters_to_parameter_vector(
        C1Stick_1_mu=mu,
        C1Stick_1_lambda_par=lambda_par,
        S4SphereGaussianPhaseApproximation_1_diameter=diameter,
        partial_volume_0=f_0,
        partial_volume_1=f_1,
        partial_volume_2=f_2,
    )

    print("Simulating VERDICT data", flush=True)
    Signals_VERDICT = VERDICT_mod.simulate_signal(acq_scheme, Parameters_VERDICT_dmipy)
    Signals_VERDICT = add_noise(Signals_VERDICT)

    # Add Cartesian parameters
    _, Parameters_VERDICT = add_cartesian(VERDICT_mod, Parameters_VERDICT_dmipy)

    Signals_VERDICT = Signals_VERDICT.astype(np.float32)
    Parameters_VERDICT = Parameters_VERDICT.astype(np.float32)
    Parameters_VERDICT = verdict_params_norm(Parameters_VERDICT)

    return Signals_VERDICT, Parameters_VERDICT


def add_noise(data, scale=0.02):
    """Add Rician noise to data"""

    data_real = data + np.random.normal(scale=scale, size=np.shape(data))
    data_imag = np.random.normal(scale=scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy


def add_cartesian(model, parameters_dmipy):
    """
    Args:
        model: dmipy model
        parameters_dmipy: dmipy parameter vector

    Returns:
        parameters_all: All parameters, including spherical and Cartesian
        parameters_cart_only: Parameters replacing spherical coordinates with Cartesian
    """

    # mu_{name} is the name of the spherical coordinates to convert
    # Find mu parameter and therefore the index of theta and phi
    mu_index = [i for i, s in enumerate(model.parameter_names) if "_mu" in s][0]
    theta_phi_index = mu_index, mu_index + 1

    # Convert to cartesian coordinates
    mu_cartesian = unitsphere2cart_Nd(parameters_dmipy[:, theta_phi_index])

    # Flip the direction of any cartesian points in the lower half of the sphere
    lower_index = mu_cartesian[:, 2] < 0
    mu_cartesian[lower_index, :] = -mu_cartesian[lower_index, :]

    # Add cartesian coordinates to the parameter array
    parameters_all = np.append(parameters_dmipy, mu_cartesian, axis=1)

    # Remove spherical coordinates ("mu") from the parameter
    parameters_cart_only = np.delete(parameters_all, theta_phi_index, axis=1)

    return parameters_all, parameters_cart_only
