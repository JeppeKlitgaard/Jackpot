from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import RNGKey


class Algorithm(eqx.Module):
    """
    Base class implementation of a lattice algorithm.
    """

    @staticmethod
    def step(rng_key: RNGKey, state: State) -> State:
        """
        Does a single algorithm step on the state and returns the resultant
        state.
        """
        raise NotImplementedError("This must be implemented in subclass.")

    @staticmethod
    def sweep(rng_key: RNGKey, state: State) -> State:
        """
        Does a single algorithm sweep step on the state and returns the
        resultant state.

        For local lattice algorithm the sweep is implemented by iterating
        over all lattice sites with the step function.
        """
        raise NotImplementedError("This must be implemented in subclass.")
