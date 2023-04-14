from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

if TYPE_CHECKING:
    from ising.state import State


def get_cluster_linkage_factors(
    state: State,
    spins: Float[Array, "*dims"],
    neighbours: Float[Array, "ndim *dims"],
) -> Float[Array, "ndim *dims"]:
    """
    Calculates the Hamiltonian Delta by only considering nearest neighbours.
    This is much more efficient than calculating the Hamiltonian for each
    Metropolis step.

    Not that interaction coefficients are JIT compile-static and thus any
    Hamiltonian contributions with interaction coefficients that are zero
    are automatically discarded during JIT tree-shaking.

    Introduction of if statements to do manual tree-shaking would at best
    slow down tracing process and at worst lead to slower runtimes.

    See: http://arxiv.org/abs/2007.08593
    """
    # i = spins
    # j = neighbours
    link_factors = jnp.zeros_like(neighbours)

    # J - bilinear exchange energy (nearest neighbour)
    link_factors += state.env.interaction_bilinear * spins * neighbours

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    link_factors += state.env.interaction_biquadratic * (spins**2) * (neighbours**2)

    # L - Calculate bicubic exchange energy (nearest neighbour)
    link_factors += state.env.interaction_bicubic * (
        (spins**2) * neighbours + spins * (neighbours**2)
    )

    link_factors *= -2.0
    link_factors *= state.env.beta

    # In case we're in a high spin system we need to nuke the ones that are
    # partially aligned
    is_same = spins == neighbours
    link_factors *= is_same

    return link_factors
