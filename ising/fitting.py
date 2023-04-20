import numpy as np
from lmfit import Model
from lmfit.models import ConstantModel


def _exponential_saturation(x, amplitude, decay):
    return amplitude * (1.0 - np.exp(-(x) / decay))


SaturatingExponential = Model(_exponential_saturation) + ConstantModel()
