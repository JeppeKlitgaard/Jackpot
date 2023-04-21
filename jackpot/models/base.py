from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
from jaxtyping import Array, Float, UInt

from jackpot.typing import TSpin

if TYPE_CHECKING:
    from jackpot.state import State


class Model(eqx.Module):
    """
    Represents a lattice model.
    """

    # Possible spin states
    spin_states: tuple[float, ...] = eqx.static_field()

    def get_magnetisation_density(self, state: State) -> float:
        """
        Returns the magnetisation density of the state.
        """
        raise NotImplementedError("This must be implemented on subclass.")

    def get_hamiltonian(self, state: State) -> float:
        """
        Returns the full Hamiltonian of the state.
        """
        raise NotImplementedError("This must be implemented on subclass.")

    def get_hamiltonian_delta(
        self,
        state: State,
        idx: UInt[Array, "a"],
        trial_spin: TSpin,
    ) -> float:
        """
        Returns a delta of the Hamiltonian given replacement of the spin
        at `idx` with `trial_spin`.
        """
        raise NotImplementedError("This must be implemented on subclass.")

    def get_cluster_linkage_factors(
        self,
        state: State,
        spins: Float[Array, "*dims"],
        neighbours: Float[Array, "ndim *dims"],
    ) -> Float[Array, "ndim *dims"]:
        """
        Returns the cluster linkage factors for use in the clustering algorithm
        implemented in `jackpot.algorithms.cluster`
        """
        raise NotImplementedError("This must be implemented on subclass.")
