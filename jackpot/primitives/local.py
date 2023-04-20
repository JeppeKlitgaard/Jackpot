from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array, random
from jax import ensure_compile_time_eval as compile_time
from jaxtyping import Float, Int, UInt

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey, TIndex, TIndexArray, TShape


def get_random_point_idx(rng_key: RNGKey, shape: TShape) -> TIndex:
    with compile_time():
        dim = len(shape)

        minvals = np.zeros(dim)
        maxvals = np.asarray(shape)

    idx = random.randint(key=rng_key, shape=(dim,), minval=minvals, maxval=maxvals)

    return tuple(idx)


def apply_boundary_conditions(
    *, state: State, idxs: Int[Array, "a b"]
) -> UInt[Array, "a b"]:
    """
    Applies boundary conditions to an array of indices.

    Assumes square array.

    This resolves the indices using the boundary condition such that the
    returned indices are ∈ ℕ ∪ {0}.
    """
    # Note: We assume square array
    side_length = state.spins.shape[0]
    upper_bound = side_length - 1

    # More readable but requires concrete idxs:
    # idxs = idxs.at[idxs > upper_bound].add(-side_length)
    idxs = jnp.where(idxs <= upper_bound, idxs, idxs - side_length)

    return idxs


def get_nearest_neighbour_idxs(
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
    spins = []
    for idx in idxs:
        # For periodic we assume BC already applied
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
