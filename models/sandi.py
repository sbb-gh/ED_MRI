import numpy as np
import torch
from scipy.special import jv
from scipy.optimize import brentq


GAMMA = 2.0 * np.pi * 42.57747892e6


def _sphere_root_equation(x):
    """
    Equation defining the dimensionless sphere roots.
    """
    return jv(1.5, x) / x - jv(2.5, x)


def _calculate_sphere_roots(n_roots=20):
    """
    Calculate the dimensionless sphere roots used in the
    Gaussian Phase Distribution approximation.
    """

    x = np.linspace(
        1e-3,
        100.0,
        100000,
    )

    y = _sphere_root_equation(x)

    roots = []

    for i in range(len(x) - 1):

        if y[i] * y[i + 1] < 0:

            root = brentq(
                _sphere_root_equation,
                x[i],
                x[i + 1],
            )

            if (
                len(roots) == 0
                or abs(root - roots[-1]) > 1e-6
            ):
                roots.append(root)

            if len(roots) == n_roots:
                break

    return np.asarray(roots)


SPHERE_ROOTS = torch.tensor(
    _calculate_sphere_roots(20),
    dtype=torch.float64,
)


def _calculate_b(
    delta,
    Delta,
    G,
):
    """
    PGSE b-value.

    Parameters
    ----------
    delta : torch.Tensor
        Gradient duration [s].
    Delta : torch.Tensor
        Gradient separation [s].
    G : torch.Tensor
        Gradient strength [T/m].

    Returns
    -------
    torch.Tensor
        b-value [s/m^2].
    """

    return (
        GAMMA
        * G
        * delta
    ) ** 2 * (
        Delta
        - delta / 3.0
    )


def _stick_signal(
    b,
    D,
):
    """
    Powder-averaged stick signal.

    b : s/m^2
    D : m^2/s
    """

    x = torch.sqrt(
        torch.clamp(
            b * D,
            min=1e-14,
        )
    )

    return (
        np.sqrt(np.pi)
        / (2.0 * x)
        * torch.erf(x)
    )


def _ball_signal(
    b,
    D,
):
    """
    Isotropic Gaussian compartment.
    """

    return torch.exp(
        -b * D
    )


def _sphere_signal(
    delta,
    Delta,
    G,
    radius,
    D_soma,
    roots=SPHERE_ROOTS,
):
    """
    Restricted sphere signal using the GPD approximation.

    Parameters
    ----------
    delta : torch.Tensor
        Gradient duration [s].

    Delta : torch.Tensor
        Gradient separation [s].

    G : torch.Tensor
        Gradient strength [T/m].

    radius : torch.Tensor
        Sphere radius [m].

    D_soma : torch.Tensor
        Intra-soma diffusivity [m^2/s].
    """

    alpha = roots / radius

    alpha2 = alpha**2

    a2D = (
        alpha2[None, :]
        * D_soma
    )

    d = delta[:, None]
    Dlt = Delta[:, None]

    exp1 = torch.exp(
        -a2D
        * (Dlt - d)
    )

    exp2 = torch.exp(
        -a2D * d
    )

    exp3 = torch.exp(
        -a2D * Dlt
    )

    exp4 = torch.exp(
        -a2D
        * (Dlt + d)
    )

    bracket = (
        2.0 * d
        - (
            2.0
            + exp1
            - 2.0 * exp2
            - 2.0 * exp3
            + exp4
        )
        / a2D
    )

    denominator = (
        alpha2
        * radius**2
        - 2.0
    )

    series = (
        alpha[None, :] ** -4
        / denominator[None, :]
        * bracket
    ).sum(dim=1)

    exponent = (
        -2.0
        * (
            GAMMA * G
        ) ** 2
        / D_soma
        * series
    )

    return torch.exp(
        exponent
    )


def sandi_signal(
    theta,
    delta,
    Delta,
    G,
):
    """
    SANDI signal.

    theta =
        [
            f_neurite,
            f_soma,
            D_neurite,
            D_extra,
            R_soma,
        ]

    Units
    -----
    diffusivity : m^2/s
    radius      : m
    """

    (
        f_neurite,
        f_soma,
        D_neurite,
        D_extra,
        R_soma,
    ) = theta

    f_extra = (
        1.0
        - f_neurite
        - f_soma
    )

    D_soma = torch.tensor(
        3.0e-9,
        dtype=theta.dtype,
        device=theta.device,
    )

    b = _calculate_b(
        delta,
        Delta,
        G,
    )

    S_neurite = _stick_signal(
        b,
        D_neurite,
    )

    S_extra = _ball_signal(
        b,
        D_extra,
    )

    S_soma = _sphere_signal(
        delta,
        Delta,
        G,
        R_soma,
        D_soma,
    )

    return (
        f_neurite * S_neurite
        + f_soma * S_soma
        + f_extra * S_extra
    )