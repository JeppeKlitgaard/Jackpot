"""
Implementation of the Wolff cluster algorithm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from jax import random

from jackpot.algorithms.cluster import (
    ClusterAlgorithm,
    ClusterSelection,
    ClusterSolution,
)
from jackpot.primitives.local import get_random_point_idx

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey


class WolffAlgorithm(ClusterAlgorithm):
    """
    Implementation of the Wolff algorithm.

    Notably features support for the Metropolis-Hastings-like acceptance
    modification used to enable working with systems that interact with the
    environment. This could, for example, be an external magnetic field or
    some other spin-lattice interaction that cannot be wrapped up in the
    clustering step.
    """

    def step(self, rng_key: RNGKey, state: State) -> State:
        raise NotImplementedError("Wolff is a cluster algorithm and can only sweep")

    def sweep(self, rng_key: RNGKey, state: State) -> State:
        point_key, flip_key = random.split(key=rng_key, num=2)

        # First step is solving for the clusters
        solution = ClusterSolution.clusterise_state(rng_key=rng_key, state=state)

        # Next we find a cluster and select it
        seed_idx = get_random_point_idx(rng_key=point_key, shape=state.shape)
        current_spin = state.spins[tuple(seed_idx)]

        selection = ClusterSelection.from_seed_idx(
            cluster_solution=solution, seed_idx=seed_idx
        )

        return self.do_flip(
            rng_key=flip_key,
            selection=selection,
            state=state,
            current_spin=current_spin,
        )
