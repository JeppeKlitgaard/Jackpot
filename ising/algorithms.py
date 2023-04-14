"""
Contains algorithms for propagating a State in time.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array, lax, random
from jax import ensure_compile_time_eval as compile_time
from jaxtyping import Bool, Float, UInt, Int

from ising.primitives2 import (
    get_hamiltonian,
    get_hamiltonian_delta,
    get_nearest_neighbour_idxs,
    get_random_point_idx,
    get_spins,
    get_trial_spin,
    set_spin,
)
from ising.typing import TSpins

if TYPE_CHECKING:
    from ising.state import State
    from ising.typing import RNGKey, TSpin

    TAcceptFunc = Callable[
        [RNGKey, Float[Array, ""], Float[Array, ""]], Bool[Array, ""]
    ]


@eqx.filter_jit
def metropolis_hastings_accept(
    rng_key: RNGKey, beta: Float[Array, ""], delta: Float[Array, ""]
) -> Bool[Array, ""]:
    if_energy_lower = lambda: True

    def if_energy_higher() -> bool:
        x = random.uniform(rng_key)
        threshold = jnp.exp(-beta * delta)
        return threshold > x

    return lax.cond(delta < 0, if_energy_lower, if_energy_higher)


@eqx.filter_jit
def glauber_accept(
    rng_key: RNGKey, beta: Float[Array, ""], delta: Float[Array, ""]
) -> Bool[Array, ""]:
    x = random.uniform(rng_key)
    threshold = 1.0 / (1.0 + jnp.exp(beta * delta))
    acceptance = threshold > x

    return acceptance


@eqx.filter_jit
def local_update_step(
    accept_func: TAcceptFunc,
    idx: UInt[Array, ""],
    rng_key: RNGKey,
    state: State,
) -> State:
    spin_key, accept_key = random.split(rng_key, 2)

    current_spin = state.spins[tuple(idx)]
    trial_spin = get_trial_spin(
        rng_key=spin_key, state=state, current_spin=current_spin
    )
    H_delta = get_hamiltonian_delta(state=state, idx=idx, trial_spin=trial_spin)

    accept = accept_func(accept_key, state.env.beta, H_delta)
    new_spin: TSpin = jnp.where(accept, trial_spin, current_spin)
    state = set_spin(state=state, idx=idx, new_spin=new_spin)

    # Update steps
    where = lambda s: s.steps
    state = eqx.tree_at(where, state, state.steps + 1)

    return state


@eqx.filter_jit
def local_update_sweep(
    accept_func: TAcceptFunc, rng_key: RNGKey, state: State
) -> State:
    with compile_time():
        idxs = tuple(np.ndindex(state.spins.shape))

    keys = random.split(key=rng_key, num=state.spins.size)

    # For loop that carries out individual steps
    def body_fun(i: int, state: State) -> State:
        out: State = local_update_step(
            accept_func,
            jnp.asarray(idxs)[i],
            keys[i],
            state,
        )
        return out

    state = lax.fori_loop(0, len(idxs), body_fun, state)

    return state


@eqx.filter_jit
def metropolis_hastings_step(rng_key: RNGKey, state: State) -> State:
    point_key, update_key = random.split(rng_key, 2)
    idx = get_random_point_idx(rng_key=point_key, shape=state.shape)

    return local_update_step(
        accept_func=metropolis_hastings_accept,
        idx=idx,
        rng_key=update_key,
        state=state,
    )


@eqx.filter_jit
def metropolis_hastings_sweep(rng_key: RNGKey, state: State) -> State:
    return local_update_sweep(
        accept_func=metropolis_hastings_accept,
        rng_key=rng_key,
        state=state,
    )


@eqx.filter_jit
def glauber_step(rng_key: RNGKey, state: State) -> State:
    point_key, update_key = random.split(rng_key, 2)
    idx = get_random_point_idx(rng_key=point_key, shape=state.shape)

    return local_update_step(
        accept_func=glauber_accept,
        idx=idx,
        rng_key=update_key,
        state=state,
    )


@eqx.filter_jit
def glauber_sweep(rng_key: RNGKey, state: State) -> State:
    return local_update_sweep(
        accept_func=glauber_accept,
        rng_key=rng_key,
        state=state,
    )


def get_cluster_linkage_factor(
    state: State,
    spin_magnitude: TSpin,  # Effective abs(delta_spin)
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
    link_factor: ScalarFloat = 0.0

    spin_magnitude_sq = jnp.square(spin_magnitude)

    # J - bilinear exchange energy (nearest neighbour)
    link_factor += 2 * state.env.interaction_bilinear * spin_magnitude.sum()

    # K - Calculate biquadratic exchange energy (nearest neighbour)
    link_factor += 2 * state.env.interaction_biquadratic * spin_magnitude_sq.sum()

    # L - Calculate bicubic exchange energy (nearest neighbour)
    link_factor += (
        state.env.interaction_bicubic * (spin_magnitude_sq * spin_magnitude).sum()
    )

    return link_factor


# @eqx.filter_jit
# def wolff_sweep(rng_key: RNGKey, state: State) -> State:
#     spins = state.spins

#     rng_key, seed_key, spin_key, accept_key = random.split(rng_key, 4)
#     seed_idx = get_random_point_idx(rng_key=seed_key, shape=state.shape)
#     seed_spin = spins[tuple(seed_idx)]

#     new_spin = get_trial_spin(rng_key=spin_key, state=state, current_spin=seed_spin)

#     # Calculate threshold value (P_add)
#     with compile_time():
#         spin_magnitude = np.abs(new_spin - seed_spin)
#         link_factor = get_cluster_linkage_factor(
#             state=state, spin_magnitude=spin_magnitude
#         )
#         threshold = 1.0 - np.exp(-link_factor * state.env.beta)

#     # Flip spins as we go on trial state and determine whether to manifest
#     # later
#     trial_state = set_spin(state=state, idx=seed_idx, new_spin=new_spin)

#     unvisited = deque([seed_idx])

#     def dbg(*args, **kwargs):
#         # print(*args, **kwargs)
#         ...

#     steps = 0

#     while unvisited:
#         site = unvisited.pop()
#         neighbour_idxs = get_nearest_neighbour_idxs(state=trial_state, idx=site)
#         neighbours = get_spins(state=trial_state, idxs=neighbour_idxs)

#         filtered = list(
#             filter(lambda x: x[1] == seed_spin, zip(neighbour_idxs, neighbours))
#         )
#         if not filtered:
#             continue

#         filter_neighbour_idxs, _ = zip(*filtered)

#         rng_key, *add_keys = random.split(rng_key, len(neighbours))

#         for add_key, neighbour_idx in zip(
#             add_keys, filter_neighbour_idxs
#         ):
#             x = random.uniform(add_key)
#             if threshold > x:
#                 trial_state = set_spin(
#                     state=trial_state, idx=neighbour_idx, new_spin=new_spin
#                 )
#                 unvisited.appendleft(neighbour_idx)
#                 steps += 1

#     # Now determine if we manifest trial state using Metropolis-Hastings
#     # acceptance function
#     H_delta = get_hamiltonian(trial_state) - get_hamiltonian(state)

#     accept = metropolis_hastings_accept(accept_key, state.env.beta, H_delta)
#     state = lax.cond(accept, lambda: trial_state, lambda: state)


#     return state
# @eqx.filter_jit
def wolff_sweep(rng_key: RNGKey, state: State) -> State:
    return state

    spins = state.spins

    rng_key, seed_key, spin_key, accept_key = random.split(rng_key, 4)
    seed_idx = get_random_point_idx(rng_key=seed_key, shape=state.shape)
    seed_spin = spins[tuple(seed_idx)]

    new_spin = get_trial_spin(rng_key=spin_key, state=state, current_spin=seed_spin)
    # Calculate threshold value (P_add)
    with compile_time():
        spin_magnitude = np.abs(new_spin - seed_spin)
        link_factor = get_cluster_linkage_factor(
            state=state, spin_magnitude=spin_magnitude
        )
        threshold = 1.0 - np.exp(-link_factor * state.env.beta)

    # Flip spins as we go on trial state and determine whether to manifest
    # later
    trial_state = set_spin(state=state, idx=seed_idx, new_spin=new_spin)

    unvisited = deque([seed_idx])

    def dbg(*args, **kwargs):
        # print(*args, **kwargs)
        ...

    steps = 0

    while unvisited:
        site = unvisited.pop()
        neighbour_idxs = get_nearest_neighbour_idxs(state=trial_state, idx=site)
        neighbours = get_spins(state=trial_state, idxs=neighbour_idxs)

        filtered = list(
            filter(lambda x: x[1] == seed_spin, zip(neighbour_idxs, neighbours))
        )
        if not filtered:
            continue

        filter_neighbour_idxs, _ = zip(*filtered)

        rng_key, *add_keys = random.split(rng_key, len(neighbours))

        for add_key, neighbour_idx in zip(add_keys, filter_neighbour_idxs):
            x = random.uniform(add_key)
            if threshold > x:
                trial_state = set_spin(
                    state=trial_state, idx=neighbour_idx, new_spin=new_spin
                )
                unvisited.appendleft(neighbour_idx)
                steps += 1

    # Now determine if we manifest trial state using Metropolis-Hastings
    # acceptance function
    H_delta = get_hamiltonian(trial_state) - get_hamiltonian(state)

    accept = metropolis_hastings_accept(accept_key, state.env.beta, H_delta)
    state = lax.cond(accept, lambda: trial_state, lambda: state)

    return state


import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (3, 3)
plt.rcParams["figure.dpi"] = 100

from jax.scipy.signal import convolve
import matplotlib.patches as mpatches

# from scipy.signal import convolve


def show(arr, title):
    plt.figure()
    im = plt.imshow(arr, interpolation=None)
    plt.title(title)

    # Legends
    values = np.unique(arr.ravel())
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [
        mpatches.Patch(color=colors[i], label=f"{values[i]}")
        for i in range(len(values))
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def convolve_with_wrapping(array, conv_array):
    assert array.ndim == conv_array.ndim
    pad_sizes = np.asarray(conv_array.shape) // 2

    padded_shape = tuple(
        ax + pad_size * 2 for ax, pad_size in zip(array.shape, pad_sizes)
    )
    padded_array = jnp.zeros(padded_shape, dtype=array.dtype)
    # Add array to padded array
    middle_slice = tuple(
        slice(pad_size, dim + pad_size) for pad_size, dim in zip(pad_sizes, array.shape)
    )
    padded_array = padded_array.at[middle_slice].set(array)

    # Add 'sides'
    for i in range(array.ndim):
        slicer1_from = tuple(
            slice(dim + pad_size - pad_size, dim + 2 * pad_size - pad_size)
            if i == j
            else slice(None)
            for j, (pad_size, dim) in enumerate(zip(pad_sizes, array.shape))
        )
        slicer1_to = tuple(
            slice(0, pad_size) if i == j else slice(None)
            for j, (pad_size, dim) in enumerate(zip(pad_sizes, array.shape))
        )

        slicer2_from = tuple(
            slice(pad_size, 2 * pad_size) if i == j else slice(None)
            for j, (pad_size, dim) in enumerate(zip(pad_sizes, array.shape))
        )
        slicer2_to = tuple(
            slice(dim + pad_size, dim + 2 * pad_size) if i == j else slice(None)
            for j, (pad_size, dim) in enumerate(zip(pad_sizes, array.shape))
        )
        padded_array.at[slicer1_to].set(padded_array[slicer1_from])
        padded_array.at[slicer2_to].set(padded_array[slicer2_from])

    print(padded_array.dtype)
    print(conv_array.dtype)

    convolved_padded = convolve(padded_array, conv_array, mode="same")
    print(convolved_padded.dtype)
    convolved = convolved_padded[middle_slice].astype(array.dtype)

    # print(padded_array)
    # print(convolved_padded)
    # print(convolved)

    return convolved


class ClusterSolverState(eqx.Module):
    do_not_visit: Bool[Array, "..."]
    updated: Bool[Array, "..."]

    neighbour_kernel: Int[Array, "..."]

    randoms: ...
    random_accept: ...

    rng_key: RNGKey
    threshold: float

    steps: int
    is_done: bool

    @staticmethod
    def should_continue(cluster_state: Self) -> bool:
        return cluster_state.is_done == False

    @staticmethod
    def evolve(cluster_state: Self) -> Self:
        cs = cluster_state
        new_key, randoms_key = random.split(cluster_state.rng_key)

        # Find neighbours
        neighbour_finder = cs.updated.astype(int)
        neighbour_finder = convolve_with_wrapping(neighbour_finder, cs.neighbour_kernel)
        neighbours = jnp.clip(neighbour_finder, 0, 2).astype(bool)

        # Find candidates
        candidates = neighbours & ~cs.do_not_visit & ~cs.updated

        # Generate random numbers for probabilistic addition to cluster
        randoms = random.uniform(key=randoms_key, shape=candidates.shape)
        random_accept = cs.threshold > randoms

        # Find new cluster members
        new_cluster_members = candidates & random_accept

        # Reflect updates on other arrays
        updated = cs.updated | new_cluster_members

        # We have visited all states this time around, thus
        # no unvisited states if we didn't add new members
        # in which case we are done
        is_done = new_cluster_members.sum() == 0

        return ClusterSolverState(
            do_not_visit=cs.do_not_visit,
            updated=updated,
            neighbour_kernel=cs.neighbour_kernel,
            rng_key=new_key,
            threshold=cs.threshold,
            steps=cs.steps + new_cluster_members.sum(),
            is_done=is_done,
            randoms=randoms,
            random_accept=random_accept,
        )


# @eqx.filter_jit
def wolff_sweep(rng_key: RNGKey, state: State) -> State:
    print("Recompiled")
    spins = state.spins

    rng_key, seed_key, spin_key, accept_key, random_key, cluster_key = random.split(
        rng_key, 6
    )
    seed_idx = get_random_point_idx(rng_key=seed_key, shape=state.shape)
    seed_spin = spins[tuple(seed_idx)]

    new_spin = get_trial_spin(rng_key=spin_key, state=state, current_spin=seed_spin)

    # Calculate threshold value (P_add)
    spin_magnitude = jnp.abs(new_spin - seed_spin)
    link_factor = get_cluster_linkage_factor(state=state, spin_magnitude=spin_magnitude)
    threshold = 1.0 - jnp.exp(-link_factor * state.env.beta)
    print(threshold)

    # Construct initial cluster state
    updated = jnp.zeros(spins.shape, dtype=bool)
    updated = updated.at[tuple(seed_idx)].set(True)
    do_not_visit = spins != seed_spin
    random_accept = jnp.ones(spins.shape, dtype=bool)
    conv = jnp.array(
        [
            [0, 1, 0],
            [1, -99999, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )

    cluster_state = ClusterSolverState(
        do_not_visit=do_not_visit,
        updated=updated,
        neighbour_kernel=conv,
        rng_key=cluster_key,
        threshold=threshold,
        randoms=jnp.zeros(spins.shape, dtype=float),
        random_accept=random_accept,
        steps=0,
        is_done=False,
    )

    show(cluster_state.updated, "updated")
    cluster_state = lax.while_loop(
        cluster_state.should_continue, cluster_state.evolve, cluster_state
    )
    show(cluster_state.do_not_visit, "do_not_visit")
    show(cluster_state.updated, "updated")
    show(cluster_state.randoms, "randoms")
    show(cluster_state.random_accept, "random_accept")



    print(cluster_state)

    # Do not visit these sites because spin alignment is wrong
    do_not_visit = trial_spins != seed_spin

    # These sites have been visited and trial state updated
    updated = jnp.zeros(shape, dtype=bool)
    updated = updated.at[tuple(seed_idx)].set(True)


    conv = jnp.array(
        [
            [0, 1, 0],
            [1, -99999, 1],
            [0, 1, 0],
        ],
        dtype=int,
    )

    neighbour_finder = updated.astype(int)
    print(neighbour_finder.dtype)
    neighbour_finder = convolve_with_wrapping(neighbour_finder, conv)
    neighbours = np.clip(neighbour_finder, 0, 2)

    # updated_idxs = jnp.asarray(jnp.nonzero(updated, size=size, fill_value=-1)).T
    # print("updated_idxs")
    # print(updated_idxs)

    # # We can't use convolution because it does not support periodic boundary
    # # conditions :(
    # vget_nearest_neighbour_idxs = eqx.filter_vmap(in_axes=(None, 0))(
    #     get_nearest_neighbour_idxs
    # )
    # raw_neighbour_idxs = vget_nearest_neighbour_idxs(state, updated_idxs)
    # neighbour_idxs = rearrange(raw_neighbour_idxs, "batch site idx -> (batch site) idx")

    # print("neighbour_idxs")
    # print(neighbour_idxs)
    # flat_neighbour_idxs = jnp.ravel_multi_index(neighbour_idxs.T, shape, mode="wrap")
    # neighbours = jnp.zeros(shape, dtype=bool)
    # neighbours_flat = neighbours.reshape(-1)
    # neighbours_flat = neighbours_flat.at[flat_neighbour_idxs].set(True)
    # neighbours = neighbours_flat.reshape(shape)

    # show(neighbours, "neighbours")
    candidates = neighbours & ~do_not_visit & ~updated

    randoms = random.uniform(key=random_key, shape=shape)
    random_accept = threshold > randoms

    new_cluster_members = candidates & random_accept
    show(new_cluster_members, "new_cluster_members")

    # Reflect updates on other arrays
    updated = updated & new_cluster_members
    trial_spins = jnp.where(new_cluster_members, new_spin, trial_spins)

    show(state.spins, "state.spins")
    show(trial_spins, "trial_spins")
    # # Initialise
    # status = Status.NO_VISIT * jnp.ones(shape, dtype=int)
    # # Mark all correctly aligned spins as unvisited
    # status = jnp.where(trial_state.spins == seed_spin, Status.UNVISITED, status)
    # # Mark seed site as visited
    # status = status.at[tuple(seed_idx)].set(Status.VISITED)
    # status = status.at[tuple([seed_idx[0] + 1, seed_idx[1]])].set(Status.VISITED)

    print(randoms)
    print(mask)

    print("ASD")
    visited_idx = jnp.transpose(jnp.asarray(jnp.nonzero(status == Status.VISITED)))
    print(visited_idx)

    vget_nearest_neighbour_idxs = eqx.filter_vmap(in_axes=(None, 0))(
        get_nearest_neighbour_idxs
    )
    raw_to_visit_idxs = vget_nearest_neighbour_idxs(state, visited_idx)
    to_visit_idxs = rearrange(raw_to_visit_idxs, "batch site idx -> (batch site) idx")

    # for idx
    print(to_visit_idxs)
    print(to_visit_idxs.shape)

    plt.figure()
    plt.imshow(status)
    print(status)

    return state

    # def cluster_updater(state:)

    while unvisited:
        site = unvisited.pop()
        neighbour_idxs = get_nearest_neighbour_idxs(state=trial_state, idx=site)
        neighbours = get_spins(state=trial_state, idxs=neighbour_idxs)

        filtered = list(
            filter(lambda x: x[1] == seed_spin, zip(neighbour_idxs, neighbours))
        )
        if not filtered:
            continue
        return state

        filter_neighbour_idxs, _ = zip(*filtered)

        rng_key, *add_keys = random.split(rng_key, len(neighbours))

        for add_key, neighbour_idx in zip(add_keys, filter_neighbour_idxs):
            x = random.uniform(add_key)
            if threshold > x:
                trial_state = set_spin(
                    state=trial_state, idx=neighbour_idx, new_spin=new_spin
                )
                unvisited.appendleft(neighbour_idx)
                steps += 1

    # Now determine if we manifest trial state using Metropolis-Hastings
    # acceptance function
    H_delta = get_hamiltonian(trial_state) - get_hamiltonian(state)

    accept = metropolis_hastings_accept(accept_key, state.env.beta, H_delta)
    state = lax.cond(accept, lambda: trial_state, lambda: state)

    return state


#     @eqx.filter_jit
# def metropolis_hastings_accept(
#     rng_key: RNGKey, beta: Float[Array, ""], delta: Float[Array, ""]
# ) -> Bool[Array, ""]:
#     x = random.uniform(rng_key)
#     threshold = jnp.exp(-beta * delta)
#     acceptance = threshold > x

#     return acceptance


#     return state
