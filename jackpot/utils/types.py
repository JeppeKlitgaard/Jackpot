from typing import Any

import jax.core
from jax.errors import ConcretizationTypeError


def lists_to_tuples(lst: list[Any]) -> tuple[Any, ...]:
    """
    Recursively transform nested lists to nested tuples.
    """
    return tuple(lists_to_tuples(i) if isinstance(i, list) else i for i in lst)


def is_tracer(obj: Any) -> bool:
    """
    Returns True if `obj` is a JAX Tracer.
    Otherwise returns False.
    """
    return isinstance(obj, jax.core.Tracer)


def assert_concrete(obj: Any, obj_name: str = "") -> None:
    """
    Raises an appropriate error if `obj` is not a concrete value.
    """
    if is_tracer(obj):
        if not obj_name:
            obj_name = "<Unnamed>"

        _msg = f"Object `{obj_name}` needs to be concrete, but was a Tracer. "
        raise ConcretizationTypeError(obj, _msg)
