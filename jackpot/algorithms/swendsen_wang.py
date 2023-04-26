"""
Implementation of the Swendsen-Wang cluster algorithm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import lax, random
from jaxtyping import Array, Bool

from jackpot.algorithms.cluster import (
    ClusterAlgorithm,
    ClusterSelection,
    ClusterSolution,
)

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey


class SwendsenWangSweeper(eqx.Module):
    """
    This class implements the sweeper mechanism in the Swendsen-Wang algorithm.
    It ensures we sweep all available clusters and probabilistically "flip"
    them as dictated by the Swendsen-Wang algorithm.

    The `to_sweep` boolean mask keeps track of which clusters have been visited.
    """

    rng_key: RNGKey
    state: State

    cluster_solution: ClusterSolution
    algorithm: ClusterAlgorithm = eqx.static_field()

    to_sweep: Bool[Array, "*dims"]

    @classmethod
    @eqx.filter_jit
    def sweep(cls, rng_key: RNGKey, state: State, algorithm: ClusterAlgorithm) -> State:
        """
        Performs a whole sweep by iteratively applying a single Swendsen-Wang
        step.
        """
        cluster_key, sweep_key = random.split(key=rng_key, num=2)

        to_sweep = jnp.ones_like(state.spins, dtype=bool)
        cluster_solution = ClusterSolution.clusterise_state(
            rng_key=cluster_key, state=state
        )

        sweeper = cls(
            rng_key=sweep_key,
            state=state,
            cluster_solution=cluster_solution,
            algorithm=algorithm,
            to_sweep=to_sweep,
        )

        sweeper = lax.while_loop(sweeper.should_continue, sweeper.step, sweeper)

        return sweeper.state

    @staticmethod
    @eqx.filter_jit
    def should_continue(sweeper: SwendsenWangSweeper) -> bool:
        """
        Determines whether the sweeping is done.
        """
        return sweeper.to_sweep.sum() != 0

    @staticmethod
    @eqx.filter_jit
    def step(sweeper: SwendsenWangSweeper) -> SwendsenWangSweeper:
        """
        Performs a single sweep step.
        """
        should_flip_key, flip_key, new_key = random.split(key=sweeper.rng_key, num=3)

        # First true entry, except if none found, then (0, 0)
        seed_idx = jnp.asarray(
            jnp.unravel_index(sweeper.to_sweep.argmax(), sweeper.to_sweep.shape)
        )
        selection = ClusterSelection.from_seed_idx(
            cluster_solution=sweeper.cluster_solution, seed_idx=seed_idx
        )

        # Update the states we have already swept
        to_sweep = sweeper.to_sweep & ~selection.selected

        # Flip selection with 50% chance
        should_flip = random.uniform(should_flip_key) >= 0.50

        do_not_flip = lambda _k, s: s

        def do_flip(key: RNGKey, state: State) -> State:
            current_spin = sweeper.state.spins[tuple(seed_idx)]
            return sweeper.algorithm.do_flip(
                rng_key=key,
                selection=selection,
                state=state,
                current_spin=current_spin,
            )

        state = lax.cond(should_flip, do_flip, do_not_flip, flip_key, sweeper.state)

        return SwendsenWangSweeper(
            rng_key=new_key,
            state=state,
            cluster_solution=sweeper.cluster_solution,
            algorithm=sweeper.algorithm,
            to_sweep=to_sweep,
        )


class SwendsenWangAlgorithm(ClusterAlgorithm):
    """
    Implementation of the Swendsen-Wang algorithm.

    Clusterises the state and then randomises each cluster with a probability
    of 50%.

    Notably features support for the Metropolis-Hastings-like acceptance
    modification used to enable working with systems that interact with the
    environment. This could, for example, be an external magnetic field or
    some other spin-lattice interaction that cannot be wrapped up in the
    clustering step.
    """

    def step(self, rng_key: RNGKey, state: State) -> State:
        raise NotImplementedError(
            "Swendsen-Wang is a cluster algorithm and can only sweep"
        )

    def sweep(self, rng_key: RNGKey, state: State) -> State:
        return SwendsenWangSweeper.sweep(rng_key=rng_key, state=state, algorithm=self)
