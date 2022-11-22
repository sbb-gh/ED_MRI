import numpy as np
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.signal_models import gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.utils.utils import unitsphere2cart_Nd


def noddi_model():

    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    zeppelin = gaussian_models.G2Zeppelin()
    watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

    # fix parameters and tortuosity constraints
    watson_dispersed_bundle.set_tortuous_parameter(
        "G2Zeppelin_1_lambda_perp", "C1Stick_1_lambda_par", "partial_volume_0"
    )
    watson_dispersed_bundle.set_equal_parameter(
        "G2Zeppelin_1_lambda_par", "C1Stick_1_lambda_par"
    )
    watson_dispersed_bundle.set_fixed_parameter("G2Zeppelin_1_lambda_par", 1.7e-9)

    NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
    # fix isotropic diffusivity
    NODDI_mod.set_fixed_parameter("G1Ball_1_lambda_iso", 3e-9)

    return NODDI_mod


def noddi(nsamples, acq_scheme):

    NODDI_mod = noddi_model()

    mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(nsamples, 2))
    odi = np.random.uniform(low=0.01, high=0.99, size=nsamples)
    f_0 = np.random.uniform(low=0.01, high=0.99, size=nsamples)
    f_1 = 1 - f_0
    f_bundle = np.random.uniform(low=0.01, high=0.99, size=nsamples)

    # big parameter vector for simulate_signal
    Parameters_NODDI_dmipy = NODDI_mod.parameters_to_parameter_vector(
        SD1WatsonDistributed_1_SD1Watson_1_mu=mu,
        SD1WatsonDistributed_1_SD1Watson_1_odi=odi,
        SD1WatsonDistributed_1_partial_volume_0=f_bundle,
        partial_volume_0=f_0,
        partial_volume_1=f_1,
    )

    Signals_NODDI = NODDI_mod.simulate_signal(acq_scheme, Parameters_NODDI_dmipy)
    # NODDI_mod.parameter_names
    Signals_NODDI = add_noise(Signals_NODDI)

    # add cartesian parameters parameters_NODDI are the parameters to learn
    Parameters_NODDI_all, Parameters_NODDI = add_cartesian(
        NODDI_mod, Parameters_NODDI_dmipy
    )

    Signals_NODDI = Signals_NODDI.astype(np.float32)
    Parameters_NODDI = Parameters_NODDI.astype(np.float32)
    Parameters_NODDI_dmipy = Parameters_NODDI_dmipy.astype(np.float32)
    Parameters_NODDI_all = Parameters_NODDI_all.astype(np.float32)

    return (
        Signals_NODDI,
        Parameters_NODDI,
        NODDI_mod,
        acq_scheme,
        Parameters_NODDI_dmipy,
        Parameters_NODDI_all,
    )


def verdict_model():
    # fixed to verdict value
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=0.9e-9)
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()

    VERDICT_mod = MultiCompartmentModel(models=[sphere, ball, stick])
    # VERDICT_mod.parameter_names

    # verdict limits
    VERDICT_mod.set_fixed_parameter("G1Ball_1_lambda_iso", 0.9e-9)
    VERDICT_mod.set_parameter_optimization_bounds(
        "C1Stick_1_lambda_par", [3.05e-9, 10e-9]
    )
    VERDICT_mod.parameter_ranges

    return VERDICT_mod


def verdict_params_norm(Parameters_VERDICT):
    print("Normalizing means of Parameters_VERDICT")
    Parameters_VERDICT[:, 0] = Parameters_VERDICT[:, 0] * (10**5) * 0.5
    Parameters_VERDICT[:, 1] = Parameters_VERDICT[:, 1] * (10**8)
    return Parameters_VERDICT


def verdict(nsamples, acq_scheme, parameters=None):

    VERDICT_mod = verdict_model()

    # random parameters with sensible upper and lower bounds
    mu = np.random.uniform(low=[0, -np.pi], high=[np.pi, np.pi], size=(nsamples, 2))
    lambda_par = np.random.uniform(low=1e-9, high=10e-9, size=nsamples)  # in m^2/s
    diameter = np.random.uniform(low=0.01e-06, high=20e-06, size=nsamples)
    f_0 = np.random.uniform(low=0.01, high=0.99, size=nsamples)
    f_1 = np.random.uniform(low=0.01, high=0.99 - f_0, size=nsamples)
    f_2 = 1 - f_0 - f_1

    # put into a big parameter vector that can be passed into simulate_signal
    Parameters_VERDICT_dmipy = VERDICT_mod.parameters_to_parameter_vector(
        C1Stick_1_mu=mu,
        C1Stick_1_lambda_par=lambda_par,
        S4SphereGaussianPhaseApproximation_1_diameter=diameter,
        partial_volume_0=f_0,
        partial_volume_1=f_1,
        partial_volume_2=f_2,
    )

    Signals_VERDICT = VERDICT_mod.simulate_signal(acq_scheme, Parameters_VERDICT_dmipy)
    Signals_VERDICT = add_noise(Signals_VERDICT)

    # add cartesian parameters parameters_VERDICT are the parameters to learn
    Parameters_VERDICT_all, Parameters_VERDICT = add_cartesian(
        VERDICT_mod, Parameters_VERDICT_dmipy
    )

    Signals_VERDICT = Signals_VERDICT.astype(np.float32)
    Parameters_VERDICT = Parameters_VERDICT.astype(np.float32)
    Parameters_VERDICT_all = Parameters_VERDICT_all.astype(np.float32)
    Parameters_VERDICT_dmipy = Parameters_VERDICT_dmipy.astype(np.float32)

    Parameters_VERDICT = verdict_params_norm(Parameters_VERDICT)

    return (
        Signals_VERDICT,
        Parameters_VERDICT,
        VERDICT_mod,
        acq_scheme,
        Parameters_VERDICT_dmipy,
        Parameters_VERDICT_all,
    )


def add_noise(data, scale=0.02):
    data_real = data + np.random.normal(scale=scale, size=np.shape(data))
    data_imag = np.random.normal(scale=scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy


def add_cartesian(model, Parameters_dmipy):
    # model is dmipy model, Parameters_dmipy is dmipy parameter_vector,
    # mu_name is the name of the spherical coordinates to convert
    # find where the mu parameter is, and therefore the index of theta and phi
    mu_index = [i for i, s in enumerate(model.parameter_names) if "_mu" in s][0]
    theta_phi_index = mu_index, mu_index + 1
    # convert to cartesian coordinates
    mu_cartesian = unitsphere2cart_Nd(Parameters_dmipy[:, theta_phi_index])
    # flip the direction of any cartesian points in the lower half of the sphere
    lower_index = mu_cartesian[:, 2] < 0
    mu_cartesian[lower_index, :] = -mu_cartesian[lower_index, :]

    # add the cartesian coordinates to the parameter array
    Parameters_all = np.append(Parameters_dmipy, mu_cartesian, axis=1)
    # remove the spherical coordinates (i.e. "mu") from the main parameter array (this is what you want to learn)
    Parameters_cart_only = np.delete(Parameters_all, theta_phi_index, axis=1)

    return Parameters_all, Parameters_cart_only
