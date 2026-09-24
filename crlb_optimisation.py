import numpy as np
import torch

from scipy.optimize import brentq
from scipy.special import jv

from models.sandi import sandi_signal 

# ============================================================
# Generic CRLB optimisation
# ============================================================

def _fisher_information(
    model,
    theta,
    acquisition_params,
    parameter_scales,
    sigma,
):
    """
    Fisher information for one parameter set.
    """

    def forward(t):
        return model(
            t,
            **acquisition_params,
        )

    J = torch.autograd.functional.jacobian(
        forward,
        theta,
        create_graph=True,
    )

    J_scaled = (
        J
        * parameter_scales[None, :]
    )

    return (
        J_scaled.T
        @ J_scaled
    ) / sigma**2


def _crlb_objective(
    model,
    theta_grid,
    acquisition_params,
    parameter_scales,
    sigma,
):
    """
    Mean A-optimal CRLB across parameter sets.
    """

    losses = []

    for theta in theta_grid:

        F = _fisher_information(
            model=model,
            theta=theta,
            acquisition_params=acquisition_params,
            parameter_scales=parameter_scales,
            sigma=sigma,
        )

        reg = (
            1e-8
            * torch.eye(
                F.shape[0],
                dtype=F.dtype,
                device=F.device,
            )
        )

        crlb = torch.linalg.inv(
            F + reg
        )

        losses.append(
            torch.trace(crlb)
        )

    return torch.stack(
        losses
    ).mean()




def optimise_crlb_protocol(
    model_name,
    n_measurements,
    snr,
    n_parameter_sets=32,
    n_iterations=100,
    lr=0.03,
    seed=42,
):
    """
    Jointly optimise a complete acquisition protocol using CRLB.
    """

    model_name = model_name.upper()

    configs = {
        "ADC": _get_adc_config,
        "T1INV": _get_t1_config,
        "SANDI": _get_sandi_config,
    }

    if model_name not in configs:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(configs)}"
        )

    config = configs[model_name](
        n_measurements=n_measurements,
        n_parameter_sets=n_parameter_sets,
        seed=seed,
    )

    raw_params = config["create_raw_protocol"]()

    optimiser = torch.optim.Adam(
        raw_params,
        lr=lr,
    )

    sigma = 1 / snr

    for iteration in range(
        n_iterations
    ):
        optimiser.zero_grad()

        acquisition_params = (
            config["decode_protocol"](
                *raw_params
            )
        )

        loss = _crlb_objective(
            model=config["model"],
            theta_grid=config["theta_grid"],
            acquisition_params=acquisition_params,
            parameter_scales=config[
                "parameter_scales"
            ],
            sigma=sigma,
        )

        loss.backward()

        optimiser.step()

        if iteration % 50 == 0:
            print(
                f"{model_name} "
                f"{iteration:4d}: "
                f"CRLB={loss.item():.5g}"
            )

    acquisition_params = (
        config["decode_protocol"](
            *raw_params
        )
    )

    return config["format_output"](
        acquisition_params
    )

# ============================================================
# ADC
# ============================================================

def _get_adc_config(
    n_measurements,
    n_parameter_sets,
    seed,
):

    min_b = 0.0
    max_b = 5.0

    min_D = 0.1
    max_D = 3.0


    def model(
        theta,
        b,
    ):
        """
        theta = [S0, D]
        """

        S0 = theta[0]
        D = theta[1]

        return (
            S0
            * torch.exp(
                -b * D
            )
        )


    rng = np.random.default_rng(
        seed
    )

    theta_grid = []

    for _ in range(
        n_parameter_sets
    ):

        D = rng.uniform(
            min_D,
            max_D,
        )

        theta_grid.append(
            torch.tensor(
                [
                    1.0,
                    D,
                ],
                dtype=torch.float64,
                requires_grad=True,
            )
        )


    parameter_scales = torch.tensor(
        [
            1.0,    # S0
            1.0,    # ADC
        ],
        dtype=torch.float64,
    )


    def create_raw_protocol():

        return [
            torch.randn(
                n_measurements,
                dtype=torch.float64,
                requires_grad=True,
            )
        ]


    def decode_protocol(
        raw_b,
    ):

        b = (
            min_b
            + torch.sigmoid(
                raw_b
            )
            * (
                max_b
                - min_b
            )
        )

        return {
            "b": b,
        }


    def format_output(
        acquisition_params,
    ):

        return (
            acquisition_params["b"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


    return dict(
        model=model,
        theta_grid=theta_grid,
        parameter_scales=parameter_scales,
        create_raw_protocol=create_raw_protocol,
        decode_protocol=decode_protocol,
        format_output=format_output,
    )

# ============================================================
# T1 inversion recovery
# ============================================================

def _get_t1_config(
    n_measurements,
    n_parameter_sets,
    seed,
):

    min_ti = 0.1
    max_ti = 7.0

    min_T1 = 0.1
    max_T1 = 7.0

    TR = 7.0


    def model(
        theta,
        ti,
    ):
        """
        theta = [S0, T1]
        """

        S0 = theta[0]
        T1 = theta[1]

        return (
            S0
            * (
                1
                - 2
                * torch.exp(
                    -ti / T1
                )
                + torch.exp(
                    torch.tensor(
                        -TR,
                        dtype=ti.dtype,
                        device=ti.device,
                    )
                    / T1
                )
            )
        )


    rng = np.random.default_rng(
        seed
    )

    theta_grid = []

    for _ in range(
        n_parameter_sets
    ):

        T1 = rng.uniform(
            min_T1,
            max_T1,
        )

        theta_grid.append(
            torch.tensor(
                [
                    1.0,
                    T1,
                ],
                dtype=torch.float64,
                requires_grad=True,
            )
        )


    parameter_scales = torch.tensor(
        [
            1.0,    # S0
            2.0,    # T1
        ],
        dtype=torch.float64,
    )


    def create_raw_protocol():

        return [
            torch.randn(
                n_measurements,
                dtype=torch.float64,
                requires_grad=True,
            )
        ]


    def decode_protocol(
        raw_ti,
    ):

        ti = (
            min_ti
            + torch.sigmoid(
                raw_ti
            )
            * (
                max_ti
                - min_ti
            )
        )

        return {
            "ti": ti,
        }


    def format_output(
        acquisition_params,
    ):

        return (
            acquisition_params["ti"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


    return dict(
        model=model,
        theta_grid=theta_grid,
        parameter_scales=parameter_scales,
        create_raw_protocol=create_raw_protocol,
        decode_protocol=decode_protocol,
        format_output=format_output,
    )


# ============================================================
# SANDI
# ============================================================


def _generate_sandi_parameter_grid(
    n=32,
    seed=42,
):
    """
    Generate representative SANDI tissue parameters.
    """

    rng = np.random.default_rng(
        seed
    )

    params = []

    while len(params) < n:

        f_neurite = rng.uniform(
            0.1,
            0.7,
        )

        f_soma = rng.uniform(
            0.05,
            0.5,
        )

        if (
            f_neurite
            + f_soma
            >= 0.95
        ):
            continue

        D_neurite = (
            rng.uniform(
                0.5,
                3.0,
            )
            * 1e-9
        )

        D_extra = (
            rng.uniform(
                0.5,
                3.0,
            )
            * 1e-9
        )

        R_soma = (
            rng.uniform(
                2.0,
                12.0,
            )
            * 1e-6
        )

        params.append([
            f_neurite,
            f_soma,
            D_neurite,
            D_extra,
            R_soma,
        ])

    return [
        torch.tensor(
            p,
            dtype=torch.float64,
            requires_grad=True,
        )
        for p in params
    ]

def _decode_sandi_protocol(
    raw_delta,
    raw_Delta,
    raw_G,
    delta_range=(
        5e-3,
        40e-3,
    ),
    Delta_range=(
        15e-3,
        80e-3,
    ),
    G_range=(
        0.01,
        0.300,
    ),
    min_gap=2e-3,
):
    """
    Convert unconstrained optimisation variables into a
    physically valid PGSE protocol.
    """

    delta_min, delta_max = (
        delta_range
    )

    Delta_min, Delta_max = (
        Delta_range
    )

    G_min, G_max = (
        G_range
    )

    delta = (
        delta_min
        + torch.sigmoid(
            raw_delta
        )
        * (
            delta_max
            - delta_min
        )
    )

    minimum_Delta = torch.maximum(
        torch.full_like(
            delta,
            Delta_min,
        ),
        delta + min_gap,
    )

    Delta = (
        minimum_Delta
        + torch.sigmoid(
            raw_Delta
        )
        * (
            Delta_max
            - minimum_Delta
        )
    )

    G = (
        G_min
        + torch.sigmoid(
            raw_G
        )
        * (
            G_max
            - G_min
        )
    )

    return (
        delta,
        Delta,
        G,
    )


def _get_sandi_config(
    n_measurements,
    n_parameter_sets,
    seed,
):

    theta_grid = (
        _generate_sandi_parameter_grid(
            n=n_parameter_sets,
            seed=seed,
        )
    )

    parameter_scales = torch.tensor(
        [
            0.5,
            0.5,
            1e-9,
            1e-9,
            5e-6,
        ],
        dtype=torch.float64,
    )


    def create_raw_protocol():

        return [
            torch.zeros(
                n_measurements,
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.zeros(
                n_measurements,
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.randn(
                n_measurements,
                dtype=torch.float64,
                requires_grad=True,
            ),
        ]


    def decode_protocol(
        raw_delta,
        raw_Delta,
        raw_G,
    ):

        (
            delta,
            Delta,
            G,
        ) = _decode_sandi_protocol(
            raw_delta,
            raw_Delta,
            raw_G,
        )

        return {
            "delta": delta,
            "Delta": Delta,
            "G": G,
        }


    def format_output(
        acquisition_params,
    ):

        return np.column_stack([
            acquisition_params[
                "delta"
            ]
            .detach()
            .cpu()
            .numpy()
            * 1e3,

            acquisition_params[
                "Delta"
            ]
            .detach()
            .cpu()
            .numpy()
            * 1e3,

            acquisition_params[
                "G"
            ]
            .detach()
            .cpu()
            .numpy()
            * 1e3,
        ]).astype(
            np.float32
        )


    return {
        "model": sandi_signal,
        "theta_grid": theta_grid,
        "parameter_scales":
            parameter_scales,
        "create_raw_protocol":
            create_raw_protocol,
        "decode_protocol":
            decode_protocol,
        "format_output":
            format_output,
    }