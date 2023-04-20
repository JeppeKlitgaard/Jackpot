from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

if TYPE_CHECKING:
    from jackpot.typing import T, TShape


def flatten_ensamble(tree: PyTree[Any], shape: TShape) -> PyTree[Any]:
    """
    Flattens an ensample given a vectorisation shape.
    """

    def mapper(item: Any) -> Any:
        if not eqx.is_array(item):
            return item

        assert item.shape[: len(shape)] == shape, "bad shape"

        remaining_shape = item.shape[len(shape) :]
        new_shape = (-1, *remaining_shape)

        return jnp.reshape(item, new_shape)

    return jtu.tree_map(mapper, tree)


def unensamble(tree: PyTree[T], shape: TShape) -> list[T]:
    """
    Disassemble an ensample made using a series of Equinox
    filter transformations.

    See: https://github.com/patrick-kidger/equinox/issues/313
    See: https://docs.kidger.site/equinox/tricks/#ensembling
    """
    flat_tree = flatten_ensamble(tree=tree, shape=shape)
    flat_idxs = jnp.arange(np.prod(shape))

    parts = []
    for idx in flat_idxs:
        filter_ = lambda x: x[idx] if eqx.is_array(x) else x
        part = jtu.tree_map(filter_, flat_tree)
        parts.append(part)

    return parts
