"""
Holds common logic that local lattice algorithms share.
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

from jackpot.primitives.state import get_trial_spin
from jackpot.primitives.utils import set_spin

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey, TSpin

    TAcceptFunc = Callable[
        [RNGKey, Float[Array, ""], Float[Array, ""]], Bool[Array, ""]
    ]


def local_update_step(
    accept_func: TAcceptFunc,
    idx: UInt[Array, ""],
    rng_key: RNGKey,
    state: State,
) -> State:
    """
    Shared logic for all local algorithms.
    """
    spin_key, accept_key = random.split(rng_key, 2)

    current_spin = state.spins[tuple(idx)]
    trial_spin = get_trial_spin(
        rng_key=spin_key, state=state, current_spin=current_spin
    )
    H_delta = state.model.get_hamiltonian_delta(
        state=state, idx=idx, trial_spin=trial_spin
    )

    accept = accept_func(accept_key, state.beta, H_delta)
    new_spin: TSpin = jnp.where(accept, trial_spin, current_spin)
    state = set_spin(state=state, idx=idx, new_spin=new_spin)

    # Update steps
    where = lambda s: s.steps
    state = eqx.tree_at(where, state, state.steps + 1)

    return state


def local_update_sweep(
    accept_func: TAcceptFunc, rng_key: RNGKey, state: State
) -> State:
    """
    Shared logic for all local algorithms.
    """
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
