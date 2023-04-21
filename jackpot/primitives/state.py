from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import ensure_compile_time_eval as compile_time
from jax import lax, random

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey, TSpin


# TODO: Move into model class.
def get_trial_spin(*, rng_key: RNGKey, state: State, current_spin: float) -> TSpin:
    """
    Returns a new spin that is guaranteed to be different to `current_spin`.
    This is achieved in an efficient and deterministic manner using the `lax.switch`
    primitive with switch branches being determined and lowered at
    compile time.
    """
    with compile_time():
        spin_states = state.model.spin_states
        spin_states_arr = np.asarray(spin_states)

        current_spin_idx = jnp.where(
            spin_states_arr == current_spin, size=1, fill_value=len(spin_states)
        )[0][0]

        # In order to construct branch of different candidates as a function
        # of current spin we need to use lax.switch statement
        # We an array holding all other possible spins than the current spin
        # We make such an array for each possible current spin
        # We need to include random.choice in branches because size of
        # candidates varies (we could have current_spin not in spins,
        # in which case we return all possible spins)
        def branch_maker(i):
            def branch(rng_key: RNGKey):
                if i == len(spin_states):
                    candidates = spin_states_arr
                else:
                    candidates = np.delete(spin_states_arr, i)

                return random.choice(key=rng_key, a=candidates)

            return branch

        branches = [branch_maker(i) for i in range(len(spin_states) + 1)]

    return lax.switch(current_spin_idx, branches, rng_key)
