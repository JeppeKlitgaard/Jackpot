"""
Contains algorithms for propagating a State in time.
"""
from __future__ import annotations
from ising.primitives2 import get_random_point_idx, get_hamiltonian_delta
from jax import random
import numpy as np
from jax import Array, lax
import jax.numpy as jnp
import equinox as eqx
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ising.typing import RNGKey, TSpin, TSpins
    from ising.state import State



@eqx.filter_jit
def metropolis_hastings_move(rng_key: RNGKey, state: State) -> State:
    point_key, spin_key, boltzmann_key = random.split(rng_key, 3)
    idx = get_random_point_idx(rng_key=point_key, shape=state.shape)

    current_spin = state.spins[tuple(idx)]
    trial_spin = random.choice(key=spin_key, a=np.asarray(state.env.spin_states))

    H_delta = get_hamiltonian_delta(state=state, idx=idx, trial_spin=trial_spin)

    if_energy_lower = lambda: trial_spin

    def if_energy_higher() -> TSpin:
        new_spins: Array = lax.cond(
            jnp.exp(-state.env.beta * H_delta) > random.uniform(boltzmann_key),
            lambda: trial_spin,
            lambda: current_spin,
        )
        return new_spins

    new_spin: TSpin = lax.cond(H_delta < 0, if_energy_lower, if_energy_higher)
    new_spins: TSpins = state.spins.at[tuple(idx)].set(new_spin)
    where = lambda tree: tree.spins
    new_state: State = eqx.tree_at(where, state, new_spins)

    return new_state

@eqx.filter_jit
def metropolis_hastings_moves(
    steps: int,
    rng_key: RNGKey,
    state: State,
) -> State:
    keys = random.split(rng_key, steps)

    def body_fun(i: int, state: State) -> State:
        out: State = metropolis_hastings_move(
            keys[i],
            state,
        )
        return out

    result: State = lax.fori_loop(0, steps, body_fun, state)
    return result
