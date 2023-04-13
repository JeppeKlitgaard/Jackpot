
from typing import TypeVar

from jax import Array
from jax.random import PRNGKeyArray
from jaxtyping import Float, Int, UInt

T = TypeVar("T")

ScalarFloat = Float[Array, ""]
ScalarInt = Int[Array, ""]
ScalarUInt = UInt[Array, ""]

TSpin = Float[Array, ""]
TSpins = Float[Array, "..."]

RNGKey = Array | PRNGKeyArray

TShape = tuple[int, ...]
TIndex = tuple[int, ...]
