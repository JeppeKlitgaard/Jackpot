from typing import Self

import equinox as eqx

from ising.typing import TShape
from ising.utils.tree import unensamble


class EnsamblableModule(eqx.Module):
    """
    Equinox modules that has additional support for ensambling.
    """

    @property
    def vectorisation_shape(self) -> TShape:
        raise NotImplementedError("This should be overridden!")

    @property
    def is_vectorised(self) -> bool:
        return bool(self.vectorisation_shape)

    def unensamble(self) -> list[Self]:
        return unensamble(self, self.vectorisation_shape)
