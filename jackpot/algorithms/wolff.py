"""
Implementation of the Wolff cluster algorithm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import lax, random

from jackpot.algorithms.cluster import (
    ClusterAlgorithm,
    ClusterSelection,
    ClusterSolution,
)
from jackpot.algorithms.metropolis_hastings import (
    metropolis_hastings_accept,
)
from jackpot.primitives.local import get_random_point_idx
from jackpot.primitives.state import get_trial_spin

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
        point_key, spin_key, accept_key = random.split(key=rng_key, num=3)

        # First step is solving for the clusters
        solution = ClusterSolution.clusterise_state(rng_key=rng_key, state=state)

        # Next we find a cluster and select it
        seed_idx = get_random_point_idx(rng_key=point_key, shape=state.shape)
        current_spin = state.spins[tuple(seed_idx)]

        selection = ClusterSelection.from_seed_idx(
            cluster_solution=solution, seed_idx=seed_idx
        )

        # Set the cluster to our a new spin on our trial state
        trial_spin = get_trial_spin(
            rng_key=spin_key, state=state, current_spin=current_spin
        )
        trial_spins = jnp.where(selection.selected, trial_spin, state.spins)

        where = lambda s: s.spins
        trial_state = eqx.tree_at(where, state, trial_spins)

        # Update number of steps taken
        new_steps = selection.selected.sum()
        where = lambda s: s.steps
        trial_state = eqx.tree_at(where, trial_state, trial_state.steps + new_steps)

        # Probabilistically select trial state
        # This is a standard technique when using external field or anisotropy
        # interactions.
        # It essentially adds a Metropolis-Hastings like transition probability
        # to the cluster update, which enables these dynamics in a way that
        # cannot be accomplished using link dynamics.
        if self.probabilistic_cluster_accept:
            delta_H = state.model.get_hamiltonian(
                trial_state
            ) - state.model.get_hamiltonian(state)
            accept = metropolis_hastings_accept(
                rng_key=accept_key, beta=state.beta, delta=delta_H
            )

            new_state = lax.cond(accept, lambda: trial_state, lambda: state)

        else:
            new_state = trial_state

        return new_state
