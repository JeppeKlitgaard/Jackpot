import functools as ft
from collections.abc import Callable
from typing import Any, overload

import equinox as eqx
import jax.tree_util as jtu
from equinox import module_update_wrapper
from jax import lax


class _MapWrapper(eqx.Module):
    _fun: Callable

    def __call__(self, /, *args: Any, **kwargs: Any):
        if len(kwargs):
            raise RuntimeError(
                "keyword arguments cannot be used with functions wrapped with "
                "`filter_map`"
            )
        del kwargs

        dynamic_args, static_args = eqx.partition(args, eqx.is_array)
        static_outs = None

        def to_map(dynamic_xs):
            nonlocal static_outs
            x = eqx.combine(dynamic_xs, static_args)
            out = self._fun(*x)
            dynamic_out, static_outs = eqx.partition(out, eqx.is_array)
            return dynamic_out

        dynamic_outs = lax.map(to_map, dynamic_args)
        return eqx.combine(dynamic_outs, static_outs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)


@overload
def filter_map() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...


@overload
def filter_map(
    fun: Callable[..., Any],
) -> Callable[..., Any]:
    ...


def filter_map(
    fun=None,
):
    """
    Transformation with similar signature to `filter_{pmap,vmap,...}`.

    Wraps `jax.lax.map` and is primarily useful for looping over an axis that
    cannot be `vmap`'ed due to device-side memory constraints.
    """
    if fun is None:
        ft.partial(
            filter_map,
        )
        return

    map_wrapper = _MapWrapper(
        _fun=fun,
    )

    return module_update_wrapper(map_wrapper, fun)
