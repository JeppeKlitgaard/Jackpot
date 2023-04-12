import equinox as eqx
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from ising.typing import T, TShape


def unensamble(tree: PyTree[T], shape: TShape) -> list[T]:
    """
    Disassemble an ensample made using a series of Equinox
    filter transformations.

    See: https://github.com/patrick-kidger/equinox/issues/313
    See: https://docs.kidger.site/equinox/tricks/#ensembling
    """
    filter_ = lambda x: x[idx] if eqx.is_array(x) else x

    parts = []
    for idx in np.ndindex(shape):
        part = jtu.tree_map(filter_, tree)
        parts.append(part)

    return parts