from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.signal import convolve
from scipy.ndimage import generate_binary_structure

from ising.primitives.local import get_nearest_neighbours
from ising.typing import ScalarFloat, TSpin

if TYPE_CHECKING:
    from ising.state import State


def get_magnetisation_density(state: State) -> ScalarFloat:
    magnetisation = state.env.nuclear_magnetic_moment * jnp.sum(state.spins)
    return magnetisation / jnp.size(state.spins)


def get_hamiltonian(state: State) -> ScalarFloat:
    # Find a kernel we can use with convolution
    kernel = generate_binary_structure(state.dim, 1)
    np.put(kernel, kernel.size // 2, False)

    H: ScalarFloat = jnp.asarray(0.0)
    env = state.env
    spins = state.spins
    spins_sq = jnp.square(spins)

    # J - Calculate bilinear exchange energy (nearest neighbour)
    H -= env.interaction_bilinear * (spins * convolve(spins, kernel, mode="same")).sum()

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    H -= (
        env.interaction_biquadratic
        * (spins_sq * convolve(spins_sq, kernel, mode="same")).sum()
    )

    # D - Calculate anisotropy energy
    H -= env.interaction_anisotropy * spins_sq.sum()

    # L - Calculate bicubic exchange energy (nearest neighbour)
    H -= (
        env.interaction_bicubic
        * 2
        * (spins * convolve(spins_sq, kernel, mode="same")).sum()
    )

    # H - Calculate external field energy
    H -= env.nuclear_magnetic_moment * env.interaction_external_field * spins.sum()

    return H


def get_hamiltonian_delta(
    state: State,
    idx: Array,
    trial_spin: TSpin,
) -> ScalarFloat:
    """
    Calculates the Hamiltonian Delta by only considering nearest neighbours.
    This is much more efficient than calculating the Hamiltonian for each
    Metropolis step.

    Not that interaction coefficients are JIT compile-static and thus any
    Hamiltonian contributions with interaction coefficients that are zero
    are automatically discarded during JIT tree-shaking.

    Introduction of if statements to do manual tree-shaking would at best
    slow down tracing process and at worst lead to slower runtimes.
    """
    H: ScalarFloat = 0.0

    current_spin = state.spins[tuple(idx)]
    delta_spin = trial_spin - current_spin

    neighbours = get_nearest_neighbours(
        state=state,
        idx=idx,
    )

    neighbours_sq = jnp.square(neighbours)
    delta_spin_sq = jnp.square(delta_spin)

    # J - Calculate bilinear exchange energy (nearest neighbour)
    H -= 2 * state.env.interaction_bilinear * (delta_spin * neighbours).sum()

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    H -= 2 * state.env.interaction_biquadratic * (delta_spin_sq * neighbours_sq).sum()

    # D - Calculate anisotropy energy
    H -= state.env.interaction_anisotropy * delta_spin_sq

    # L - Calculate bicubic exchange energy (nearest neighbour)
    H -= (
        state.env.interaction_bicubic
        * (delta_spin_sq * neighbours + delta_spin * neighbours_sq).sum()
    )

    # H - Calculate external field energy
    H -= (
        state.env.nuclear_magnetic_moment
        * state.env.interaction_external_field
        * delta_spin
    )

    return H
