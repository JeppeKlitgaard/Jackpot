from __future__ import annotations

from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import numpy as np
import pandas as pd
from jax import block_until_ready

from jackpot.utils.tree import flatten_ensamble, unensamble

if TYPE_CHECKING:
    from jackpot.typing import TShape

TFieldKeys = None | dict[str, "TFieldKeys"]


def _get_keys(obj: Any) -> TFieldKeys:
    if not is_dataclass(obj):
        return None

    keys = {}
    fields = dc_fields(obj)
    for field in fields:
        field_obj = getattr(obj, field.name)
        keys[field.name] = _get_keys(field_obj)

    return keys


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

    def flatten(self) -> Self:
        return flatten_ensamble(self, self.vectorisation_shape)

    def transform_recipe_filter(self, key: str) -> bool:
        return True

    @property
    def transform_recipe(self) -> list[str]:
        """
        A list of attributes (which may contain `.`s) to use when
        transforming unensambled object to a Python or Pandas object(s).
        """
        keys = _get_keys(self)

        # Use pd.json_normalize as a convenient dict flattener
        flat_keys_df = pd.json_normalize(keys, sep=".")
        recipe = flat_keys_df.columns.values.tolist()

        recipe = list(filter(self.transform_recipe_filter, recipe))

        return recipe

    def post_transform(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Can be used to apply additional transformations after initial unensambling.
        """
        return data_dict

    def to_dict(self) -> dict[str, list[Any]]:
        """
        Converts module to a dict of lists
        """
        data_dict: dict[str, Any] = {}
        size = np.prod(self.vectorisation_shape)
        flat_self = flatten_ensamble(self, self.vectorisation_shape)

        for key in self.transform_recipe:
            getter = attrgetter(key)
            value = getter(flat_self)

            if eqx.is_array(value):
                if value.size == 1:
                    # We want scalar values to be scalars, not (1,) arrays
                    value = [value.item()] * size
                else:
                    # Convert to numpy to prevent problems with df.explode
                    # later
                    value = list(np.asarray(value))
            else:
                value = [value] * size

            data_dict[key] = value

        return self.post_transform(data_dict)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.to_dict())

    def block_until_ready(self) -> None:
        block_until_ready(self)
