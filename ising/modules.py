from jaxtyping import Array, Float, UInt

from ising.module import EnsamblableModule
from ising.state import State
from ising.typing import TShape


class AutocorrelationData(EnsamblableModule):
    state: State

    steps: int
    sweeps_per_step: int

    sweeps: UInt[Array, "a"]

    energy: Float[Array, "a"]
    magnetisation_density: Float[Array, "a"]

    energy_decay_time: float
    magnetisation_density_decay_time: float

    @property
    def vectorisation_shape(self) -> TShape:
        return self.energy.shape[:-1]

    def transform_recipe_filter(self, key: str) -> bool:
        return key != "state.spins"
