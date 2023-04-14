import jax.numpy as jnp
import numpy as np
from jax import Array, random

from ising.typing import RNGKey, TShape


class EasyKey:
    """
    An easy key that can be used outside of a strictly stateless context.
    """

    def __init__(self, seed: int | Array):
        self.seed = seed
        self._key = random.PRNGKey(seed)

    @property
    def new(self) -> RNGKey:
        """
        Return a fresh RNG Key.
        """
        self._key, k = random.split(self._key)
        return k

    def news(self, num: int) -> Array:
        """
        Return `num` fresh RNG keys as an array.
        """
        self._key, *ks = random.split(self._key, num=num + 1)
        return jnp.asarray(ks)

    def shaped(self, shape: TShape) -> Array:
        size = np.prod(shape)
        keys = self.news(num=size)
        assert keys.ndim == 2
        assert keys.shape[1] == 2

        key_shape = (
            *shape,
            2,
        )
        shaped_keys = keys.reshape(key_shape)

        return shaped_keys
