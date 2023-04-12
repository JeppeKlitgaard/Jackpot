"""
This file contains 'primitives' for the `ising` package.

Primitives are jittable pure functions.
"""
from __future__ import annotations
from jax import Array
from jax import random
from jax import jit
import jax.numpy as jnp
from ising.typing import TIndex, TShape, TSpin, ScalarFloat, RNGKey
from functools import partial
from jax.scipy.signal import convolve
from scipy import constants
from scipy.ndimage import generate_binary_structure
import numpy as np

from typing import TYPE_CHECKING
from ising.types import BCMode

if TYPE_CHECKING:
    from ising.state import State

@partial(jit, static_argnames=("shape",))
def get_random_point_idx(rng_key: RNGKey, shape: TShape) -> TIndex:
    dim = len(shape)

    minvals = np.zeros(dim)
    maxvals = np.asarray(shape)

    idx = random.randint(key=rng_key, shape=(dim,), minval=minvals, maxval=maxvals)

    return tuple(idx)

def temperature_to_beta(temperature_or_temperatures: ScalarFloat) -> ScalarFloat:
    reciprocal: ScalarFloat = constants.Boltzmann * temperature_or_temperatures
    return 1.0 / reciprocal

def get_nearest_neighbours(
    *,
    state: State,
    idx: Array,
) -> Array:
    """
    Boundary condition: OOB are set to 0.
    """
    bc_mode = state.env.bc_mode
    bc_mode_value = state.env.bc_mode_value

    nearest_neighbours = []
    for n in range(state.dim):
        for delta in [1, -1]:
            selector = jnp.array(idx)
            selector = selector.at[n].add(delta)

            # Padding with zeros
            if bc_mode == BCMode.CONSTANT:
                assert bc_mode_value is not None

                selector = jnp.where(selector == -1, selector.size + 1, selector)
                neighbour = state.spins.at[tuple(selector)].get(
                    mode="fill", fill_value=bc_mode_value
                )

            elif bc_mode == BCMode.PERIODIC:
                selector = jnp.where(selector == selector.size + 1, 0, selector)
                neighbour = state.spins.at[tuple(selector)].get()

            nearest_neighbours.append(neighbour)

    nn = jnp.asarray(nearest_neighbours)
    return nn


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


def get_hamiltonian(
    state: State
) -> ScalarFloat:
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


def get_magnetisation_density(state: State) -> ScalarFloat:
    magnetisation = state.env.nuclear_magnetic_moment * jnp.sum(state.spins)
    return magnetisation / jnp.size(state.spins)
