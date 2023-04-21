from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float

from jackpot.algorithms.base import Algorithm
from jackpot.algorithms.local import local_update_step, local_update_sweep
from jackpot.primitives.local import get_random_point_idx

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey


def glauber_accept(
    rng_key: RNGKey, beta: Float[Array, ""], delta: Float[Array, ""]
) -> Bool[Array, ""]:
    """
    Accept function for Glauber Dynamics.
    Sometimes referred to as the transition function or transition probability
    function.
    """
    x = random.uniform(rng_key)
    threshold = 1.0 / (1.0 + jnp.exp(beta * delta))
    acceptance = threshold > x

    return acceptance


class GlauberAlgorithm(Algorithm):
    """
    Implementation of the Glauber algorithm.

    The Metropolis-Hastings algorithm is generally preferred over this
    algorithm, as it performs better for lattice models.
    """

    def step(self, rng_key: RNGKey, state: State) -> State:
        point_key, update_key = random.split(rng_key, 2)
        idx = get_random_point_idx(rng_key=point_key, shape=state.shape)

        return local_update_step(
            accept_func=glauber_accept,
            idx=idx,
            rng_key=update_key,
            state=state,
        )

    @staticmethod
    def sweep(self, rng_key: RNGKey, state: State) -> State:
        return local_update_sweep(
            accept_func=glauber_accept,
            rng_key=rng_key,
            state=state,
        )
