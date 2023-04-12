
from typing import TypeVar

import numpy as np
from jax import Array
from jax.random import PRNGKeyArray

T = TypeVar("T")

Float = np.float64
Integer = int

ScalarFloat = Float | float | Array
ScalarInt = Integer | Array

TSpin = Float
TSpins = Array
TFloatParam = np.float64

RNGKey = Array | PRNGKeyArray

TShape = tuple[int, ...]
TIndex = tuple[int, ...]
