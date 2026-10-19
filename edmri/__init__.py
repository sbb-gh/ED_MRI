"""
ED_MRI: Experimental design for quantitative and diffusion MRI.

Methods for optimising MRI acquisition protocols using
Cramér-Rao lower bounds and machine learning.
"""

from .optimise import optimise_experiment
from .inference import apply_trained_model
from .crlb_optimisation import optimise_crlb_protocol

__all__ = [
    "optimise_experiment",
    "apply_trained_model",
    "optimise_crlb_protocol",
]
