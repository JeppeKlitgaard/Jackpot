from __future__ import annotations

from typing import TYPE_CHECKING, Self

import equinox as eqx

from ising.utils.tree import unensamble

if TYPE_CHECKING:
    from ising.typing import TShape


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
