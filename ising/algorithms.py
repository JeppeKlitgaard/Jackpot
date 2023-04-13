"""
Contains algorithms for propagating a State in time.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, random
from jax import ensure_compile_time_eval as compile_time
from jaxtyping import Bool, Float, UInt

from ising.primitives2 import (
    get_hamiltonian_delta,
    get_nearest_neighbours,
    get_random_point_idx,
    get_spins,
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
    x = random.uniform(rng_key)
    threshold = jnp.exp(-beta * delta)
    acceptance = threshold > x

    return acceptance


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
    trial_spin = random.choice(key=spin_key, a=np.asarray(state.env.spin_states))

    H_delta = get_hamiltonian_delta(state=state, idx=idx, trial_spin=trial_spin)

    accept = accept_func(accept_key, state.env.beta, H_delta)
    if_energy_lower = lambda: trial_spin

    def if_energy_higher() -> TSpin:
        new_spins: Array = lax.cond(
            accept,
            lambda: trial_spin,
            lambda: current_spin,
        )
        return new_spins

    new_spin: TSpin = lax.cond(H_delta < 0, if_energy_lower, if_energy_higher)
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
