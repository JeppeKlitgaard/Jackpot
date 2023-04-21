from typing import TypeVar

from jax import Array
from jax.random import PRNGKeyArray
from jaxtyping import Float, UInt

T = TypeVar("T")

TSpin = Float[Array, ""]
TSpins = Float[Array, "..."]

RNGKey = Array | PRNGKeyArray

TShape = tuple[int, ...]
TIndex = tuple[int, ...]
TIndexArray = UInt[Array, "a"]
