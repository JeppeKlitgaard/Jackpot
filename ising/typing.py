
import numpy as np
from jax import Array
from jax.random import PRNGKeyArray
from typing import Literal

Float = np.float64
Integer = int

ScalarFloat = Float | float | Array
ScalarInt = Integer | Array

TSpin = Float
TSpins = Array
TFloatParam = np.float64
TBCMode = Literal["constant", "periodic"]
TMethod = Literal["metropolis-hastings", "wolff", "swendsen-wang"]

RNGKey = Array | PRNGKeyArray
