from typing import Any



def lists_to_tuples(lst: list[Any]) -> tuple[Any]:
    """
    Recursively transform nested lists to nested tuples.
    """
    return tuple(lists_to_tuples(i) if isinstance(i, list) else i for i in lst)
