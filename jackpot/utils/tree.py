from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

if TYPE_CHECKING:
    from jackpot.typing import T, TShape


def unensamble(tree: PyTree[T], shape: TShape) -> list[T]:
    """
    Disassemble an ensample made using a series of Equinox
    filter transformations.

    See: https://github.com/patrick-kidger/equinox/issues/313
    See: https://docs.kidger.site/equinox/tricks/#ensembling
    """

    parts = []
    for idx in np.ndindex(shape):
        filter_ = lambda x: x[idx] if eqx.is_array(x) else x
        part = jtu.tree_map(filter_, tree)
        parts.append(part)

    return parts
