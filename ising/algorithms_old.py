"""
Contains algorithms for propagating a State in time.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import Array, lax, random
from jax import ensure_compile_time_eval as compile_time
from jaxtyping import Bool, Float

from ising.primitives.local import get_random_point_idx
from ising.primitives.state import get_trial_spin
from ising.primitives.utils import set_spin

if TYPE_CHECKING:
    from ising.state import State
    from ising.typing import RNGKey

    TAcceptFunc = Callable[
        [RNGKey, Float[Array, ""], Float[Array, ""]], Bool[Array, ""]
    ]


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


# @eqx.filter_jit
def wolff_sweep(rng_key: RNGKey, state: State) -> State:
    print("Recompiled")
    ClusterSolver.clusterise_state(rng_key, state)
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
