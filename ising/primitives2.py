"""
This file contains 'primitives' for the `ising` package.

Primitives are jittable pure functions.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, random
from jax.scipy.signal import convolve
from jaxtyping import Float, Int, UInt
from scipy import constants
from scipy.ndimage import generate_binary_structure

from ising.types import BCMode
from ising.typing import RNGKey, ScalarFloat, TIndex, TIndexArray, TShape, TSpin

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


def apply_boundary_conditions(
    *, state: State, idxs: Int[Array, "a b"]
) -> UInt[Array, "a b"]:
    """
    Applies boundary conditions to an array of indices.

    Assumes square array.

    This resolves the indices using the boundary condition such that the
    returned indices are ∈ ℕ ∪ {0}.
    """
    bc_mode = state.env.bc_mode
    oob_idx = state.spins.size  # Guaranteed out-of-bounds

    match bc_mode:
        case BCMode.CONSTANT:
            # Set high value to mark out-of-bounds
            # More readable but requires concrete idxs:
            # idxs = idxs.at[idxs < 0].set(oob_idx)
            idxs = jnp.where(idxs >= 0, idxs, oob_idx)

        case BCMode.PERIODIC:
            # Note: We assume square array
            side_length = state.spins.shape[0]
            upper_bound = side_length - 1

            # More readable but requires concrete idxs:
            # idxs = idxs.at[idxs > upper_bound].add(-side_length)
            idxs = jnp.where(idxs <= upper_bound, idxs, idxs - side_length)

    return idxs


def get_nearest_neighbour_idxs(
    *,
    state: State,
    idx: TIndexArray,
) -> UInt[Array, "a b"]:
    """
    Retrieves the indices of the nearest neighbours of `idx`.
    """
    nn_idxs = []
    for n in range(state.dim):
        for delta in [1, -1]:
            selector = jnp.array(idx)
            selector = selector.at[n].add(delta)
            nn_idxs.append(selector)

    nn_idxs_arr = apply_boundary_conditions(state=state, idxs=jnp.asarray(nn_idxs))

    return nn_idxs_arr


def get_spins(*, state: State, idxs: UInt[Array, "a b"]) -> Float[Array, a]:
    """
    Look up a list of spins by idxs and return them.

    This assumes idxs have already had boundary conditions applied.
    """
    bc_mode = state.env.bc_mode
    bc_mode_value = state.env.bc_mode_value

    spins = []
    for idx in idxs:
        # Has potentially out-of-bounds indices after BC applied
        # These are intended as sentinel-like values
        if bc_mode == BCMode.CONSTANT:
            spin = state.spins.at[tuple(idx)].get(mode="fill", fill_value=bc_mode_value)

        # For periodic we assume BC already applied
        else:
            spin = state.spins[tuple(idx)]

        spins.append(spin)

    return jnp.asarray(spins)


def get_nearest_neighbours(
    *,
    state: State,
    idx: Array,
) -> Array:
    idxs = get_nearest_neighbour_idxs(state=state, idx=idx)
    neighbours = get_spins(state=state, idxs=idxs)

    return neighbours


def set_spin(*, state: State, idx: TIndexArray, new_spin: TSpin) -> State:
    """
    Convenience function for updating a spin on a state.
    """
    # Construct new spins array
    spins = state.spins.at[tuple(idx)].set(new_spin)

    # Update in State PyTree
    where = lambda s: s.spins
    state = eqx.tree_at(where, state, spins)

    return state


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


def get_magnetisation_density(state: State) -> ScalarFloat:
    magnetisation = state.env.nuclear_magnetic_moment * jnp.sum(state.spins)
    return magnetisation / jnp.size(state.spins)
