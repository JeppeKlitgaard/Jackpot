"""
Contains algorithms for propagating a State in time.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from enum import IntEnum, auto
from typing import TYPE_CHECKING
from einops import rearrange

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, random
from jax import ensure_compile_time_eval as compile_time
from jaxtyping import Bool, Float, UInt

from ising.primitives2 import (
    get_hamiltonian,
    get_hamiltonian_delta,
    get_nearest_neighbour_idxs,
    get_random_point_idx,
    get_spins,
    get_trial_spin,
    set_spin,
)

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

# from scipy.signal import convolve
from jax.scipy.signal import convolve
import matplotlib.patches as mpatches


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


# @eqx.filter_jit
def wolff_sweep(rng_key: RNGKey, state: State) -> State:
    print("Recompiled")
    spins = state.spins

    rng_key, seed_key, spin_key, accept_key, random_key = random.split(rng_key, 5)
    seed_idx = get_random_point_idx(rng_key=seed_key, shape=state.shape)
    seed_spin = spins[tuple(seed_idx)]

    new_spin = get_trial_spin(rng_key=spin_key, state=state, current_spin=seed_spin)

    # Calculate threshold value (P_add)
    spin_magnitude = jnp.abs(new_spin - seed_spin)
    link_factor = get_cluster_linkage_factor(state=state, spin_magnitude=spin_magnitude)
    threshold = 1.0 - jnp.exp(-link_factor * state.env.beta)

    # Flip spins as we go on trial state and determine whether to manifest
    # later
    trial_state = set_spin(state=state, idx=seed_idx, new_spin=new_spin)

    unvisited = deque([seed_idx])

    def dbg(*args, **kwargs):
        # print(*args, **kwargs)
        ...

    steps = 0

    class Status(IntEnum):
        NO_VISIT = auto()
        UNVISITED = auto()
        VISITED = auto()

    # Create cluster status array of same shape as initial state
    # 0: Do not visit because spin is wrong
    # 1: Unvisited
    # 2: Visited and randomised in trial state
    shape = state.spins.shape
    size = state.spins.size

    # Do not visit these sites because spin alignment is wrong
    do_not_visit = trial_state.spins != seed_spin

    # These sites have been visited and trial state updated
    updated = jnp.zeros(shape, dtype=bool)
    updated = updated.at[tuple(seed_idx)].set(True)

    print(do_not_visit)
    # show(do_not_visit, "do_not_visit")
    # show(updated, "updated")

    updated_idxs = jnp.asarray(jnp.nonzero(updated, size=size, fill_value=-1)).T
    print("updated_idxs")
    print(updated_idxs)
    # We can't use convolution because it does not support periodic boundary
    # conditions :(
    vget_nearest_neighbour_idxs = eqx.filter_vmap(in_axes=(None, 0))(
        get_nearest_neighbour_idxs
    )
    raw_neighbour_idxs = vget_nearest_neighbour_idxs(state, updated_idxs)
    neighbour_idxs = rearrange(raw_neighbour_idxs, "batch site idx -> (batch site) idx")

    print("neighbour_idxs")
    print(neighbour_idxs)
    flat_neighbour_idxs = jnp.ravel_multi_index(neighbour_idxs.T, shape, mode="wrap")
    neighbours = jnp.zeros(shape, dtype=bool)
    neighbours_flat = neighbours.reshape(-1)
    neighbours_flat = neighbours_flat.at[flat_neighbour_idxs].set(True)
    neighbours = neighbours_flat.reshape(shape)

    # show(neighbours, "neighbours")

    to_try = neighbours & ~do_not_visit & ~updated
    # show(to_try, "to_try")
    # # Initialise
    # status = Status.NO_VISIT * jnp.ones(shape, dtype=int)
    # # Mark all correctly aligned spins as unvisited
    # status = jnp.where(trial_state.spins == seed_spin, Status.UNVISITED, status)
    # # Mark seed site as visited
    # status = status.at[tuple(seed_idx)].set(Status.VISITED)
    # status = status.at[tuple([seed_idx[0] + 1, seed_idx[1]])].set(Status.VISITED)

    randoms = random.uniform(key=random_key, shape=shape)
    mask = threshold > randoms
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
