from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from jax import Array
from jaxtyping import Float
from scipy import constants

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import TIndexArray, TSpin


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


def beta_to_temp(
    beta: Float[Array, "*dim"], human: bool = False
) -> Float[Array, "*dim"]:
    reciprocal: Float[Array, "*dim"] = beta
    if not human:
        reciprocal = reciprocal * constants.Boltzmann

    return 1.0 / reciprocal


def temp_to_beta(
    temp: Float[Array, "*dim"], human: bool = False
) -> Float[Array, "*dim"]:
    reciprocal: Float[Array, "*dim"] = temp
    if not human:
        reciprocal = reciprocal * constants.Boltzmann

    return 1.0 / reciprocal
