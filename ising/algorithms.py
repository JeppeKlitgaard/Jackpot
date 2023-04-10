"""
Contains algorithms for propagating a State in time.
"""
from ising.state import State
from ising.primitives import get_random_point_idx2
from jax import random
from ising.typing import RNGKey
import numpy as np
from jax import jit


def metropolis_hastings_move(rng_key: RNGKey, state: State) -> State:
    point_key, spin_key, boltzmann_key = random.split(rng_key, 3)
    idx = get_random_point_idx2(rng_key=point_key, shape=state.spins.shape)
    print(state.env.spin_states)
    spin_states = np.asarray(state.env.spin_states)

    trial_spin = random.choice(key=spin_key, a=spin_states)

    print(idx)
    print(trial_spin)

