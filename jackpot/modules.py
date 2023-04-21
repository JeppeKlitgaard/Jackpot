"""
Contains Equinox modules that are more dataclass-like than
states, algorithms, or models in that they don't implement logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Array, Float, UInt

from jackpot.module import EnsamblableModule

if TYPE_CHECKING:
    from jackpot.state import State
    from jackpot.typing import TShape


class Measurement(EnsamblableModule):
    """
    Represents one or more measurements of the system taken at a particular
    step for the state with the given id.
    """

    state_id: UInt[Array, ""]
    beta: Float[Array, ""]

    steps: UInt[Array, ""]
    sweeps: UInt[Array, ""]

    energy: Float[Array, "a"]
    magnetisation_density: Float[Array, "a"]

    @property
    def transformation_shape(self) -> TShape:
        return self.energy.shape[:-1]


class AutocorrelationData(EnsamblableModule):
    """
    PyTree container of autocorrelation data.
    """

    state: State

    steps: int
    sweeps_per_step: int

    sweeps: UInt[Array, "a"]

    energy: Float[Array, "a"]
    magnetisation_density: Float[Array, "a"]

    energy_decay_time: float
    magnetisation_density_decay_time: float

    @property
    def transformation_shape(self) -> TShape:
        return self.energy.shape[:-1]

    def transform_recipe_filter(self, key: str) -> bool:
        return key != "state.spins"
