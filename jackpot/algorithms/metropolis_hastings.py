"""
Implementation of the Metropolis-Hastings algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Bool, Float

from jackpot.algorithms.base import Algorithm
from jackpot.algorithms.local import local_update_step, local_update_sweep
from jackpot.primitives.local import get_random_point_idx

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey


def metropolis_hastings_accept(
    rng_key: RNGKey, beta: Float[Array, ""], delta: Float[Array, ""]
) -> Bool[Array, ""]:
    """
    Acceptance function generally used with the Metropolis-Hastings algorithm.

    Also used as a modification on cluster algorithms to enable working
    in systems with environmental interactions such as an external magnetic
    field or anisotropy effects.
    """
    if_energy_lower = lambda: True

    def if_energy_higher() -> bool:
        x = random.uniform(rng_key)
        threshold = jnp.exp(-beta * delta)
        return threshold > x

    return lax.cond(delta < 0, if_energy_lower, if_energy_higher)


class MetropolisHastingsAlgorithm(Algorithm):
    """
    Implementation of the Metropolis-Hastings algorithm.

    Bulk of logic is actually implemented in `jackpot.algorithms.local`,
    as the algorithm shares a large portion of its implementation with the
    Glauber algorithm.
    """

    def step(self, rng_key: RNGKey, state: State) -> State:
        point_key, update_key = random.split(rng_key, 2)
        idx = get_random_point_idx(rng_key=point_key, shape=state.shape)

        return local_update_step(
            accept_func=metropolis_hastings_accept,
            idx=idx,
            rng_key=update_key,
            state=state,
        )

    def sweep(self, rng_key: RNGKey, state: State) -> State:
        return local_update_sweep(
            accept_func=metropolis_hastings_accept,
            rng_key=rng_key,
            state=state,
        )
